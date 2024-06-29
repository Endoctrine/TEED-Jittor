import jittor

jittor.flags.use_cuda = 1


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss modified in DexiNed

    targets = targets.long()
    mask = targets.float()
    num_positive = jittor.sum((mask > 0.0).float()).float()
    num_negative = jittor.sum((mask <= 0.0).float()).float()

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    inputs = jittor.sigmoid(inputs)
    cost = jittor.nn.BCELoss(mask)(inputs, targets.float())
    return l_weight * cost


# ------------ cats losses ----------
def bdrloss(prediction, label, radius):
    """
    The boundary tracing loss that handles the confusing pixels.
    """

    filt = jittor.ones([1, 1, 2 * radius + 1, 2 * radius + 1])
    filt.requires_grad = False

    bdr_pred = prediction * label
    pred_bdr_sum = label * jittor.nn.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = jittor.nn.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = jittor.nn.conv2d(prediction * (1 - label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = jittor.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * jittor.log(softmax_map)
    cost[label == 0] = 0

    return jittor.sum(cost.float().mean((1, 2, 3)))


def textureloss(prediction, label, mask_radius):
    """
    The texture suppression loss that smooths the texture regions.
    """
    filt1 = jittor.ones([1, 1, 3, 3])
    filt1.requires_grad = False
    filt2 = jittor.ones([1, 1, 2 * mask_radius + 1, 2 * mask_radius + 1])
    filt2.requires_grad = False

    pred_sums = jittor.nn.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = jittor.nn.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - jittor.greater(label_sums, 0.).float()

    loss = -jittor.log(jittor.clamp(1 - pred_sums / 9, 1e-10, 1 - 1e-10))
    loss[mask == 0] = 0

    return jittor.sum(loss.float().mean((1, 2, 3)))


def cats_loss(prediction, label, l_weight=None, args=None):
    # tracingLoss

    if l_weight is None:
        l_weight = [0., 0.]
    tex_factor, bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with jittor.no_grad():
        mask = label.clone()

        num_positive = jittor.sum((mask == 1).float()).float()
        num_negative = jittor.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0

    prediction = jittor.sigmoid(prediction)

    cost = jittor.nn.binary_cross_entropy_with_logits(
        prediction.float(), label.float(), weight=mask)
    label_w = (label != 0).float()
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4)

    return cost * args.batch_size + bdr_factor * bdrcost + tex_factor * textcost
