## Considerations
- Do we definitely want ros bags as input always? It might be simpler to have the option of bags vs standard image directories.
    - For kitti, tartanAir, ETH3D it seems easier now to just define a new mono_{dataset}.cc with an added argument for the output_dir (for the estimated trajectories). Will definitely need to think about designing the ros version though since it seems like a simpler option for data logging (maybe not 100% necessary?).

- Is it worth having a Sequence class? This can have as attributes the image directory or bag file, the timestamps, the intrinsics/extrinsics and the groundtruth trajectory. Maybe we can have a different class for a sequence using image directories and timestamps file and directly rosbags?

- SLAMSystem should have a mode for docker vs singularity
