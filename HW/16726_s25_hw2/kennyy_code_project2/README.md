# Nikhil Uday Shinde
# https://github.com/nikhilushinde/cs194-26_proj3_2.2
# Modified by Zhixuan Liu

## Usage

Run the script from the command line by providing the source and target image paths:

```bash
python masking_code.py source_image.jpg target_image.jpg
```

## How It Works

### Source Image Interaction

When the source image window opens, use the following key commands:

- **p**: Toggle polygon mode.
  - In this mode, each left-click adds a vertex to the polygon.
- **q**: Finalize the polygon and switch to center selection (requires at least three points).
- **o**: Rotate the image counterclockwise (increments of 5°).
- **i**: Rotate the image clockwise (increments of 5°).
- **=**: Increase the image scale.
- **-**: Decrease the image scale.
- **r**: Reset the drawing (clears polygon points and adjustments).
- **ESC**: Exit the window and proceed to the next step.

### Target Image Interaction

After finishing with the source image, the target image window will open:

- **Left-click**: Place the mask on the target image using the previously defined offset.
- **r**: Reset the target image (clears any pasted mask).
- **ESC**: Exit the window and complete the process.

## Output Files

After the process is complete, the tool saves three files in the current directory:

- **source_image_mask.png**: The mask created from the source image.
- **target_image_mask.png**: The mask pasted onto the target image.
- **source_image_newsource.png**: The warped version of the source image, aligned to the target image.

File names are automatically generated based on the input image names.

## Example

To run the tool with example images:

```bash
python masking_code.py ./data/source_01.jpg ./data/target_01.jpg
```

This command will open the source image window first for mask creation. After you finish drawing and selecting the center, the tool will open the target image window to paste the mask. When you exit, the output files will be saved.

NOTE: May need to manually resize images so cv2.imshow can show whole image