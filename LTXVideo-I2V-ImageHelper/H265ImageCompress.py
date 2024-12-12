import os
import subprocess
import tempfile
import logging
import numpy as np
import torch

from PIL import Image
from logging import log

#changed to h265 compression

class H265ImageCompress:
    """Encodes the input with h265 compression using a configurable CRF."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The input image tensor to be compressed and decompressed."
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 28,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "Constant Rate Factor for h265 encoding (lower values mean higher quality).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compress_and_decompress"

    CATEGORY = "image"
    DESCRIPTION = """
**Encodes the input image with h265 compression using a configurable CRF**"""
    
#   Notes from MTB nodes:

#   [!NOTE]
#   This was recommended by the creators of LTX over banodoco's discord.
#   Orginal code from [mix](https://github.com/XmYx)

#   Added notes:

#   ***Original code from https://github.com/melMass/comfy_mtb/blob/main/nodes/ltx.py***

#   Modified for my testing of LTXVideo with a h264 and h265 version

#   H.265 was recomended
#   Zeev Farbman:
#   Until we figure it out on the model / inference level, I suggest to try h265 compression for frame degradation,
#   should work reasonably well, with less apparent perceptual degradation of the first frame.

#   Another thing you will notice that at the moment, stronger degradation may create stronger scene / camera motion.
#   To control for that, fps parameter from the conditioning stage can be helpful.


    def _compress_decompress_ffmpeg(self, img_array, crf):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.png")
            output_path = os.path.join(temp_dir, "output.mp4")
            decoded_path = os.path.join(temp_dir, "decoded.png")

            Image.fromarray(img_array).save(input_path)

            encode_command = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libx265",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
                "1",
                output_path,
            ]
            subprocess.run(encode_command, capture_output=True)

            decode_command = [
                "ffmpeg",
                "-y",
                "-i",
                output_path,
                "-frames:v",
                "1",
                decoded_path,
            ]
            subprocess.run(decode_command, capture_output=True)

            decoded_img = np.array(Image.open(decoded_path))
            return decoded_img

    def compress_and_decompress(self, image, crf):
        import io

        output_images = []

        try:
            import av

            for img_tensor in image:
                img_array = img_tensor.cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img_array = img_array.copy(
                    order="C"
                )  # Ensure contiguous array

                output = io.BytesIO()

                # Encode the image to h265 with the given CRF
                container = av.open(output, mode="w", format="mp4")
                stream = container.add_stream("hevc", rate=1)
                stream.width = img_array.shape[1]
                stream.height = img_array.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": str(crf)}

                frame = av.VideoFrame.from_ndarray(img_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
                container.close()

                # Decode the video back to an image
                output.seek(0)
                container = av.open(output, mode="r", format="mp4")
                decoded_frames = []
                for frame in container.decode(video=0):
                    img_decoded = frame.to_ndarray(format="rgb24")
                    decoded_frames.append(img_decoded)
                container.close()

                if len(decoded_frames) > 0:
                    img_decoded = decoded_frames[0]
                    img_decoded = torch.from_numpy(
                        img_decoded.astype(np.float32) / 255.0
                    )
                    output_images.append(img_decoded)
                else:
                    # If decoding failed, use the original image
                    output_images.append(img_tensor)
        except ImportError:
            log.warning(
                "PyAv is not installed... Falling back to the ffmpeg cli"
            )
            for img_tensor in image:
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                decoded_img = self._compress_decompress_ffmpeg(img_array, crf)
                img_decoded = torch.from_numpy(
                    decoded_img.astype(np.float32) / 255.0
                )
                output_images.append(img_decoded)

        output_images = torch.stack(output_images).to(image.device)
        return (output_images,)

NODE_CLASS_MAPPINGS = {
    "H265ImageCompress": H265ImageCompress,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H265ImageCompress": "H.265 Image Compress",
    }
