# The Physical Structure and Evolution of Transition Region Explosive Events Observed with ESIS

Roy T. Smart, Charles C. Kankelborg, and Jacob D. Parker

## Abstract

The EUV Snapshot Imaging Spectrograph (ESIS) was launched on a NASA sounding rocket in September 2019. ESIS is a computed tomography imaging spectrograph (CTIS), which records a spectrum in every pixel across a wide field of view simultaneously, with high spatial, spectral, and temporal resolution. This snapshot capability lets ESIS capture the full spatial and spectral structure of rapidly evolving features that slit-scanning instruments such as IRIS can only sample sequentially. During its 5-minute flight, ESIS observed ~20 transition region explosive events (EEs), compact brightenings with supersonic wing enhancements, in O V 630 Å. Leveraging simultaneous imaging and spectroscopy, this presentation will characterize the physical structure and dynamical evolution of these EEs, and place them into the context of other EE observations.

## The slides on the web

https://roytsmart.github.io/spd-2026/ shows the slides with their movies
playing and the speaker notes under each one. `docs/index.html` is built from
the deck (`303_30_smart_spd-2026.pptx`, which is not kept in git) with
[pptx-to-html](https://github.com/roytsmart/pptx-to-html), pointing the movies
and figures at the full-quality ones in `figures/`:

```bash
pptx-to-html 303_30_smart_spd-2026.pptx docs \
    --title "The Physical Structure and Evolution of Transition Region Explosive Events Observed with ESIS" \
    --subtitle "SPD 2026 · the slides as they were presented, with the animations playing rather than frozen" \
    --media media1.mp4=../figures/blink.mp4 \
    --media media2.mp4=../figures/blink-channels.mp4 \
    --media media3.mp4=../figures/mart-spectra.mp4 \
    --media media4.mp4=../figures/mart-moments.mp4 \
    --media media5.mp4=../figures/cinemagraph-channels.mp4 \
    --media media6.mp4=../figures/level-4-o-v.mp4 \
    --media media7.mp4=../figures/level-4-o-v-box.mp4 \
    --media media8.mp4=../figures/level-4-event.mp4 \
    --media media9.mp4=../figures/level-4-event-history-images.mp4 \
    --media media10.mp4=../figures/level-4-event-history.mp4 \
    --media media11.mp4=../figures/level-4-event-motion.mp4 \
    --media media12.mp4=../figures/mart-scene.mp4 \
    --media image1.gif=../figures/cinemagraph-channel-3.mp4 \
    --media image2.svg=../figures/iris-ee-1.svg \
    --media image3.svg=../figures/iris-ee-3.svg \
    --media level-4-lines.mp4=../figures/level-4-lines.mp4 \
    --notes
```
