#!/bin/bash
# Unedited screen capture of display :1 (Isaac Sim GUI) during one aalto eval episode.
cd /root/aaltonomous-humanoid
F=$(~/miniforge3/envs/isaac/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$F" -hide_banner -loglevel warning -f x11grab -framerate 30 -video_size 1920x1080 -draw_mouse 0 -i :1 \
     -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart -y runs/screenrec/isaac_eval_seed0.mp4 \
     > runs/screenrec/ffmpeg.log 2>&1 &
FF=$!
sleep 3
date -u +"rec start %FT%TZ"
~/miniforge3/envs/isaac/bin/python -m aaltonomous.sim.evaluate --container ghcr.io/marcomosconi/upsala-v5:v1 \
    --team upsala --seed 0 --trials 1 --out runs/screenrec/eval > runs/screenrec/eval.log 2>&1
echo "eval exit=$?"
sleep 3
kill -INT $FF; wait $FF
date -u +"rec stop %FT%TZ"
