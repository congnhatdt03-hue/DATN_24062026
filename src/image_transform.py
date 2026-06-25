"""Image transform helpers used by GUI tools."""

import cv2


def rotate_image_keep_size(image, angle_deg, center=None):
    """Rotate an image in-place canvas size around a chosen center.

    The user-facing convention is positive angle = clockwise, so we negate the
    angle before passing it to OpenCV.
    """
    if image is None:
        return None
    if abs(float(angle_deg)) < 1e-9:
        return image.copy()

    height, width = image.shape[:2]
    if center is None:
        center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -float(angle_deg), 1.0)
    if len(image.shape) == 2:
        border_value = 0
    else:
        border_value = tuple(0 for _ in range(image.shape[2]))
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
