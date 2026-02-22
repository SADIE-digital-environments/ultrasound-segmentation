"""Segment a single ultrasound image object.

This module provides a number of functions for segmenting single ultrasound images,
extracting segmentation and textual data from the images.

**Usage:**

To segment a single ultrasound image, you can use the following code:

.. code-block:: python

   from usseg.single_image_processing import data_from_image

   # Load the ultrasound image.
   PIL_img = ...
   cv2_img = ...

   # Extract segmentation and textual data from the image.
   df, XYdata = data_from_image(PIL_img, cv2_img)

"""
# Python imports
import logging
import os
from PIL import Image

# Module imports
import matplotlib.pyplot as plt
import cv2

# Local imports
from usseg import general_functions

logger = logging.getLogger(__file__)


def data_from_image(pil_img=None, cv2_img=None, image_path=None):
    """Extract segmentation and textual data from an image.

    Transitional API (temporary):
        This function is migrating from accepting pre-loaded images
        (pil_img, cv2_img) to accepting an image file path instead.

        Old usage (still supported temporarily):
            data_from_image(PIL_image, cv2_image)

        New usage (preferred):
            data_from_image(image_path="path/to/image")

        Legacy arguments will be removed in a future update once
        dependent codes have migrated.

    Args:
        pil_img (Pillow Image object) : The image in Pillow format.
        cv2_img (cv2 Image object) : The image in cv2 format.

        image_path (str, optional) : Path to the image file. Transitional option.
            If provided, this function will judge file type and load pil_img and cv2_img internally.

    Returns:
        df (pandas dataframe) : Dataframe of extracted text.
        XYdata (list) : X and Y coordinates of the extracted segmentation.
    """

    # Guard invalid input combinations
    if image_path is not None and (pil_img is not None or cv2_img is not None):
        raise ValueError(
            "Provide either image_path OR (pil_img AND cv2_img), not both."
        )


    if image_path is not None:
        # Classify file type for downstream handling (image vs DICOM)
        ext = os.path.splitext(image_path)[1].lower()
        us_dicom = ext in (".dcm", ".dicom")
        # is_image: jpeg, png, or other common image formats
        us_image = ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
        if us_image: #Load image if image path
            pil_img = Image.open(image_path)
            cv2_img = cv2.imread(image_path)
        elif us_dicom: #Load DICOM if image path is a DICOM file
            dicom_metadata = general_functions.extract_dicom_metadata(image_path)
            PIL_image, cv2_img = general_functions.extract_doppler_from_dicom(image_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # Extracts yellow text from image
    # PIL_img , cv2_img = General_functions.upscale_both_images(PIL_img,cv2_img)
    PIL_image_RGB = pil_img.convert("RGB")  # We need RGB, so convert here. with PIL
    COL = general_functions.colour_extract_vectorized(PIL_image_RGB, [255, 255, 100], 95, 95)

    # COL = General_functions.Colour_extract(PIL_image_RGB, [255, 255, 100], 100, 100)
    text_extract_failed, df = general_functions.text_from_greyscale(cv2_img, COL)
    # Failure not really relevant to the rest of the segmentation so just logged as
    # a warning for the end user.
    if text_extract_failed:
        logger.warning("Couldn't extract text from image. Continuing...")
    else:
        logger.info("Completed colour extraction.")

    # No error handling for initial segmentation as impossible to complete segmentation
    # without segmentation mask.
    segmentation_mask, Xmin, Xmax, Ymin, Ymax = general_functions.initial_segmentation(
        input_image_obj=PIL_image_RGB
    )

    # Gets ROIS
    Left_dimensions, Right_dimensions = general_functions.define_end_rois(
        segmentation_mask, Xmin, Xmax, Ymin, Ymax
    )

    # Initialise axis containers so the functions can pass if one side fails.
    Lnumber = None
    Lpositions = None
    Rnumber = None
    Rpositions = None

    # Search for ticks and labels - Left axis
    try:
        (
            Cs,
            ROIAX,
            CenPoints,
            onY,
            BCs,
            TYLshift,
            thresholded_image,
            Side,
            Left_dimensions,
            Right_dimensions,
            ROI2,
            ROI3,
        ) = general_functions.search_for_ticks(
            cv2_img, "Left", Left_dimensions, Right_dimensions
        )
        ROIAX, Lnumber, Lpositions, ROIL = general_functions.search_for_labels(
            Cs,
            ROIAX,
            CenPoints,
            onY,
            BCs,
            TYLshift,
            Side,
            Left_dimensions,
            Right_dimensions,
            cv2_img,
            ROI2,
            ROI3,
        )
        # Validate and clean left-axis ticks/labels
        Lnumber, Lpositions = general_functions.validate_axis_ticks(
            Lnumber, Lpositions, side="Left"
        )
    except Exception:
        logger.exception("Single-image: failed Left axes search")

    # Search for ticks and labels - Right axis
    try:
        (
            Cs,
            ROIAX,
            CenPoints,
            onY,
            BCs,
            TYLshift,
            thresholded_image,
            Side,
            Left_dimensions,
            Right_dimensions,
            ROI2,
            ROI3,
        ) = general_functions.search_for_ticks(
            cv2_img, "Right", Left_dimensions, Right_dimensions
        )
        ROIAX, Rnumber, Rpositions, ROIR = general_functions.search_for_labels(
            Cs,
            ROIAX,
            CenPoints,
            onY,
            BCs,
            TYLshift,
            Side,
            Left_dimensions,
            Right_dimensions,
            cv2_img,
            ROI2,
            ROI3,
        )
        # Validate and clean right-axis ticks/labels
        Rnumber, Rpositions = general_functions.validate_axis_ticks(
            Rnumber, Rpositions, side="Right"
        )
    except Exception:
        logger.exception("Single-image: failed Right axes search")

    # Estimate a global y=0 line (in image coordinates) using whichever sides
    # are available.
    try:
        y_zero = general_functions.estimate_zero_line_y(
            left_numbers=Lnumber,
            left_positions=Lpositions,
            right_numbers=Rnumber,
            right_positions=Rpositions,
        )
        if y_zero is not None:
            logger.info(f"Single-image: estimated y=0 line at y={y_zero:.2f}")
    except Exception:
        logger.exception("Single-image: zero-line estimation failed")
        y_zero = None

    # Refine segmentation and compute top curve, passing estimated zero-line when available.
    (
        refined_segmentation_mask,
        top_curve_mask,
        top_curve_coords,
    ) = general_functions.segment_refinement(
        cv2_img, Xmin, Xmax, Ymin, Ymax, y_zero=y_zero
    )

    # Digitise using the single-axis digitisation routine.
    Xplot, Yplot, Ynought = general_functions.plot_digitized_data_single_axis(
        Rnumber, Rpositions, Lnumber, Lpositions, top_curve_coords,
    )

    # Apply text-based corrections if text extraction succeeded.
    if not text_extract_failed:
        try:
            df = general_functions.plot_correction(Xplot, Yplot, df)
        except Exception:
            logger.exception("Single-image: plot_correction failed")

    plt.close("all")
    XYdata = [Xplot, Yplot]
    return df, XYdata
