import numpy as np


HAND_SIDE_BY_CLASS = {0: "left", 1: "right", 2: "left", 3: "right"}
PRIMARY_HAND_CLASSES = frozenset((0, 1))
SIDES = ("left", "right")


def hand_side_from_yolo_class(cls_id):
    cls_id = int(cls_id)
    if cls_id == 0 or cls_id == 2:
        return "left"
    if cls_id == 1 or cls_id == 3:
        return "right"
    return None


def build_grouped_hand_candidates(
    xyxy,
    confs,
    clses,
    kpxy,
    kpc,
    track_ids,
    image_shape,
    compute_bbox_fn,
):
    image_h, image_w = image_shape[:2]
    asarray = np.asarray
    float32 = np.float32
    compute_bbox = compute_bbox_fn
    left_candidates = []
    right_candidates = []
    append_left = left_candidates.append
    append_right = right_candidates.append

    if track_ids is None:
        for idx in range(len(clses)):
            cls_id = int(clses[idx])
            if cls_id == 0 or cls_id == 2:
                side = "left"
                append = append_left
            elif cls_id == 1 or cls_id == 3:
                side = "right"
                append = append_right
            else:
                continue

            kp_xy = asarray(kpxy[idx], dtype=float32)
            kp_conf = asarray(kpc[idx], dtype=float32)
            _, hand_bbox = compute_bbox(kp_xy, image_w, image_h, type="hand")
            hand_bbox = asarray(hand_bbox, dtype=float32)

            append(
                {
                    "bbox": hand_bbox,
                    "det_bbox": asarray(xyxy[idx], dtype=float32),
                    "keypoint": kp_xy,
                    "conf": kp_conf,
                    "score": float(confs[idx]),
                    "x_center": float(0.5 * (hand_bbox[0] + hand_bbox[2])),
                    "handedness": side,
                    "cls": cls_id,
                    "track_id": None,
                }
            )
    else:
        for idx in range(len(clses)):
            cls_id = int(clses[idx])
            if cls_id == 0 or cls_id == 2:
                side = "left"
                append = append_left
            elif cls_id == 1 or cls_id == 3:
                side = "right"
                append = append_right
            else:
                continue

            kp_xy = asarray(kpxy[idx], dtype=float32)
            kp_conf = asarray(kpc[idx], dtype=float32)
            _, hand_bbox = compute_bbox(kp_xy, image_w, image_h, type="hand")
            hand_bbox = asarray(hand_bbox, dtype=float32)

            raw_track_id = track_ids[idx]
            track_id = int(raw_track_id) if raw_track_id is not None else -1
            if track_id < 0:
                track_id = None

            append(
                {
                    "bbox": hand_bbox,
                    "det_bbox": asarray(xyxy[idx], dtype=float32),
                    "keypoint": kp_xy,
                    "conf": kp_conf,
                    "score": float(confs[idx]),
                    "x_center": float(0.5 * (hand_bbox[0] + hand_bbox[2])),
                    "handedness": side,
                    "cls": cls_id,
                    "track_id": track_id,
                }
            )

    return {"left": left_candidates, "right": right_candidates}


def _copy_prev_hand(prev_hand, side, track_id):
    if prev_hand is None:
        return None

    prev_hand_get = prev_hand.get
    bbox = np.array(prev_hand["bbox"], dtype=np.float32, copy=True)
    return {
        "bbox": bbox,
        "det_bbox": np.array(prev_hand_get("det_bbox", bbox), dtype=np.float32, copy=True),
        "keypoint": prev_hand_get("keypoint"),
        "conf": prev_hand_get("conf"),
        "score": float(prev_hand_get("score", 0.0)),
        "x_center": float(0.5 * (bbox[0] + bbox[2])),
        "handedness": side,
        "cls": prev_hand_get("cls"),
        "track_id": track_id,
        "is_fallback": True,
    }


def select_grouped_hand_boxes(
    candidates_by_side,
    prev_boxes,
    prev_track_ids,
    prev_miss_counts,
    max_misses,
    hand_stable_iou,
    iou_fn,
):
    prev_boxes = prev_boxes or {}
    prev_track_ids = prev_track_ids or {}
    prev_miss_counts = prev_miss_counts or {}

    selected_boxes = {"left": {}, "right": {}}
    next_track_ids = {"left": None, "right": None}
    next_miss_counts = {"left": 0, "right": 0}
    max_misses = int(max_misses)
    hand_stable_iou = float(hand_stable_iou)

    for side in SIDES:
        candidates = candidates_by_side.get(side)
        prev_side_boxes = prev_boxes.get(side)
        prev_hand = prev_side_boxes.get("hand") if prev_side_boxes is not None else None
        prev_track_id = prev_track_ids.get(side)
        prev_miss_count = int(prev_miss_counts.get(side, 0))

        if candidates:
            prev_bbox = prev_hand["bbox"] if prev_hand is not None else None
            best_candidate = None
            best_track_match = -1
            best_iou = -2.0
            best_primary_class = -1
            best_score = -1.0

            for candidate in candidates:
                candidate_track_id = candidate.get("track_id")
                track_match = int(
                    prev_track_id is not None
                    and candidate_track_id is not None
                    and candidate_track_id == prev_track_id
                )
                iou = float(iou_fn(candidate["bbox"], prev_bbox)) if prev_bbox is not None else -1.0
                candidate_cls = candidate.get("cls")
                primary_class = int(candidate_cls == 0 or candidate_cls == 1)
                score = float(candidate.get("score", 0.0))

                if (
                    track_match > best_track_match
                    or (
                        track_match == best_track_match
                        and (
                            iou > best_iou
                            or (
                                iou == best_iou
                                and (
                                    primary_class > best_primary_class
                                    or (
                                        primary_class == best_primary_class
                                        and score > best_score
                                    )
                                )
                            )
                        )
                    )
                ):
                    best_candidate = candidate
                    best_track_match = track_match
                    best_iou = iou
                    best_primary_class = primary_class
                    best_score = score

            selected = dict(best_candidate)

            if (
                selected.get("track_id") is None
                and prev_track_id is not None
                and prev_bbox is not None
                and best_iou >= hand_stable_iou
            ):
                selected["track_id"] = prev_track_id

            selected_boxes[side]["hand"] = selected
            next_track_ids[side] = selected.get("track_id")
            continue

        if prev_hand is not None and prev_miss_count < max_misses:
            selected_boxes[side]["hand"] = _copy_prev_hand(prev_hand, side, prev_track_id)
            next_track_ids[side] = prev_track_id
            next_miss_counts[side] = prev_miss_count + 1

    return selected_boxes, next_track_ids, next_miss_counts
