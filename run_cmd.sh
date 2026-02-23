python python/scripts/continuous_map_train_test.py \
    --config    configs/mcd_config.yaml  \
    --osm       example_data/kth.osm   \
    --calib     example_data/hhs_calib.yaml  \
    --init_rel_pos 64.393 66.483 38.514   \
    --osm_origin_lat 59.348268650 \
    --osm_origin_lon 18.073204280 \
    \
    --scan-dir    example_data/kth_day_06/lidar_bin/data/ \
    --label-dir   example_data/kth_day_06/labels_predicted/ \
    --gt-dir      example_data/kth_day_06/gt_labels/  \
    --pose        example_data/kth_day_06/pose_inW.csv \
    \
    --offset 1 \
    --max-scans 1000 \
    --test-fraction 1 \
    --map-state output.bki \
    \
    --prior-delta 0.1 \
    --osm-prior-strength 0.01 \
    --seed-osm-prior true \
    --resolution 0.5 \
    --l-scale 1.0 \


    # Testing Day 09 KTH with kitti trained model (coverted labels from kitti to MCD format)
    #
    # --scan-dir  /mnt/semkitti/mcd-predictions/MCD/kth_day_09/lidar_bin/data/ \
    # --label-dir /mnt/semkitti/_scratch/ \
    # --gt-dir    /mnt/semkitti/mcd-predictions/MCD/kth_day_09/gt_labels/ \
    # --pose      /mnt/semkitti/mcd-predictions/MCD/kth_day_09/pose_inW.csv \


    # Testing Day 09 KTH with MCD trained model (coverted labels from kitti to MCD format)
    #
    # --scan-dir  /mnt/semkitti/mcd-predictions/MCD/kth_day_09/lidar_bin/data/ \
    # --label-dir /mnt/semkitti/mcd-predictions/MCD/kth_day_09/inferred_labels/cenet_mcd/ \
    # --gt-dir    /mnt/semkitti/mcd-predictions/MCD/kth_day_09/gt_labels/ \
    # --pose      /mnt/semkitti/mcd-predictions/MCD/kth_day_09/pose_inW.csv \


    # Testing the examples
    #
    # --scan-dir    example_data/kth_day_06/lidar_bin/data/ \
    # --label-dir   example_data/kth_day_06/labels_predicted/ \
    # --gt-dir      example_data/kth_day_06/gt_labels/  \
    # --pose        example_data/kth_day_06/pose_inW.csv \