# Media fixtures

`video.mp4` is a tiny synthetic red H.264 video used to verify that raw base64
and falsely image-labelled video payloads are rejected before calling the Backend.
It contains no external content.

Recreate with:

```sh
ffmpeg -f lavfi -i color=c=red:s=16x16:d=0.1 -an \
  -c:v libx264 -pix_fmt yuv420p tests/fixtures/video.mp4
```

Run the regression tests with:

```sh
uv run pytest tests/test_multimodal.py -k 'content_part_type or disguised_video or invalid_image_bytes'
```
