import argparse
import os
from typing import Any, List, Optional, Tuple
import cv2 #type: ignore
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img #type: ignore
from tqdm import tqdm #type: ignore
from keras.applications.efficientnet import preprocess_input as effnet_preprocess #type: ignore
from Effi_Res_hybrid import build_feature_extractor


MODEL_TAG = "effb4_incepres_lstm5"
PREPROCESS_TAG = "effnet_tf_preprocess"


def list_videos_from_frames(frames_dir: str) -> Tuple[List[str], List[str]]:
    video_dirs = []
    labels = []

    for class_name in sorted(os.listdir(frames_dir)):
        class_path = os.path.join(frames_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for video_id in sorted(os.listdir(class_path)):
            video_path = os.path.join(class_path, video_id)
            if not os.path.isdir(video_path):
                continue

            i_dir = os.path.join(video_path, "I")
            p_dir = os.path.join(video_path, "P")
            if os.path.isdir(i_dir) and os.listdir(i_dir):
                video_dirs.append(i_dir)
                labels.append(class_name)
            elif os.path.isdir(p_dir) and os.listdir(p_dir):
                video_dirs.append(p_dir)
                labels.append(class_name)

    return video_dirs, labels


def list_frame_paths(frame_dir: str) -> List[str]:
    frames = sorted(
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return frames


def sample_frames_from_paths(frame_paths: List[str], frames_per_video: int) -> List[str]:
    if not frame_paths:
        return []

    if len(frame_paths) <= frames_per_video:
        return frame_paths

    idxs = np.linspace(0, len(frame_paths) - 1, frames_per_video, dtype=int)
    return [frame_paths[i] for i in idxs]


def _result_has_person(result: Any) -> bool:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False

    cls_vals = getattr(boxes, "cls", None)
    names = getattr(result, "names", {}) or {}
    if cls_vals is None or len(cls_vals) == 0:
        return True

    try:
        cls_ids = [int(x) for x in cls_vals.tolist()]
    except Exception:
        return True

    person_tokens = {"person", "human", "pedestrian"}
    any_known_person = False
    for cls_id in cls_ids:
        cls_name = str(names.get(cls_id, "")).strip().lower()
        if cls_name in person_tokens:
            any_known_person = True
            break

    if any_known_person:
        return True

    # If class names are unavailable (single-class custom model), accept any detection.
    return len(names) == 0


def filter_person_frame_paths(
    frame_paths: List[str],
    yolo_model: Optional[Any],
    conf_thres: float,
    iou_thres: float,
    yolo_batch_size: int,
) -> List[str]:
    if yolo_model is None:
        return frame_paths
    if not frame_paths:
        return []

    person_frames = []
    for i in range(0, len(frame_paths), yolo_batch_size):
        batch_paths = frame_paths[i:i + yolo_batch_size]
        results = yolo_model(batch_paths, conf=conf_thres, iou=iou_thres, verbose=False)
        for path, res in zip(batch_paths, results):
            if _result_has_person(res):
                person_frames.append(path)

    return person_frames


def load_batch(frame_paths: List[str], img_size: Tuple[int, int]) -> np.ndarray:
    batch = []
    for p in frame_paths:
        img = load_img(p, target_size=img_size)
        arr = img_to_array(img)
        batch.append(arr)
    batch = np.asarray(batch, dtype=np.float32)
    # Standard preprocessing for EfficientNet/InceptionResNetV2 (scales to [-1, 1])
    return effnet_preprocess(batch)


def extract_video_features(
    model: tf.keras.Model,
    video_dirs: List[str],
    frames_per_video: int,
    img_size: Tuple[int, int],
    batch_size: int,
    yolo_model: Optional[Any] = None,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.5,
    yolo_batch_size: int = 32,
) -> np.ndarray:
    features = []

    for vdir in tqdm(video_dirs, desc="Extracting video features"):
        all_frame_paths = list_frame_paths(vdir)
        person_frame_paths = filter_person_frame_paths(
            all_frame_paths,
            yolo_model=yolo_model,
            conf_thres=yolo_conf,
            iou_thres=yolo_iou,
            yolo_batch_size=yolo_batch_size,
        )
        frame_paths = sample_frames_from_paths(person_frame_paths, frames_per_video)
        if not frame_paths:
            features.append(np.zeros((model.output_shape[-1],), dtype=np.float32))
            continue

        # Process frames in mini-batches
        frame_feats = []
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch = load_batch(batch_paths, img_size)
            feats = model(batch, training=False).numpy()
            frame_feats.append(feats)

        frame_feats = np.vstack(frame_feats)
        video_feat = frame_feats.mean(axis=0)
        features.append(video_feat)

    return np.vstack(features)


def dtcwt_45_features_from_gray(gray: np.ndarray, transform: Any) -> np.ndarray:
    gray = cv2.resize(gray, (128, 128))
    gray = gray.astype(np.float32) / 255.0
    coeffs = transform.forward(gray, nlevels=3)
    feats = []
    for level in range(len(coeffs.highpasses)):
        subband = coeffs.highpasses[level][:, :, 1]  # 45-degree oriented subband
        mag = np.abs(subband)
        feats.append(float(np.mean(mag)))
        feats.append(float(np.std(mag)))
        feats.append(float(np.sum(mag ** 2)))
    return np.asarray(feats, dtype=np.float32)


def dna_features_from_gray(gray: np.ndarray) -> np.ndarray:
    pixels = gray.flatten().astype(np.uint8)
    pair1 = (pixels >> 6) & 3
    pair2 = (pixels >> 4) & 3
    pair3 = (pixels >> 2) & 3
    pair4 = pixels & 3
    all_pairs = np.concatenate([pair1, pair2, pair3, pair4])
    counts = np.bincount(all_pairs, minlength=4).astype(np.float32)
    total = counts.sum()
    if total <= 0:
        return np.zeros((4,), dtype=np.float32)
    return counts / total


def extract_handcrafted_video_features(
    video_dirs: List[str],
    frames_per_video: int,
    yolo_model: Optional[Any],
    yolo_conf: float,
    yolo_iou: float,
    yolo_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import dtcwt #type: ignore

    transform = dtcwt.Transform2d()
    dtcwt_video_feats = []
    dna_video_feats = []

    for vdir in tqdm(video_dirs, desc="Extracting DTCWT/DNA features"):
        all_frame_paths = list_frame_paths(vdir)
        person_frame_paths = filter_person_frame_paths(
            all_frame_paths,
            yolo_model=yolo_model,
            conf_thres=yolo_conf,
            iou_thres=yolo_iou,
            yolo_batch_size=yolo_batch_size,
        )
        frame_paths = sample_frames_from_paths(person_frame_paths, frames_per_video)

        if not frame_paths:
            dtcwt_video_feats.append(np.zeros((9,), dtype=np.float32))
            dna_video_feats.append(np.zeros((4,), dtype=np.float32))
            continue

        frame_dtcwt = []
        frame_dna = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_dtcwt.append(dtcwt_45_features_from_gray(gray, transform))
            frame_dna.append(dna_features_from_gray(gray))

        if not frame_dtcwt:
            dtcwt_video_feats.append(np.zeros((9,), dtype=np.float32))
            dna_video_feats.append(np.zeros((4,), dtype=np.float32))
            continue

        dtcwt_video_feats.append(np.mean(np.vstack(frame_dtcwt), axis=0).astype(np.float32))
        dna_video_feats.append(np.mean(np.vstack(frame_dna), axis=0).astype(np.float32))

    return np.vstack(dtcwt_video_feats), np.vstack(dna_video_feats)


def weighted_fuse_features(
    f_cnn: np.ndarray,
    f_dtcwt: np.ndarray,
    f_dna: np.ndarray,
    w_cnn: float,
    w_dtcwt: float,
    w_dna: float,
) -> np.ndarray:
    w = np.asarray([w_cnn, w_dtcwt, w_dna], dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("Fusion weights must be non-negative.")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("At least one fusion weight must be > 0.")
    w = w / s

    # Normalize each modality independently, then weight and concatenate.
    f_cnn_n = l2_normalize(f_cnn.astype(np.float32))
    f_dtcwt_n = l2_normalize(f_dtcwt.astype(np.float32))
    f_dna_n = l2_normalize(f_dna.astype(np.float32))

    return np.concatenate(
        [
            w[0] * f_cnn_n,
            w[1] * f_dtcwt_n,
            w[2] * f_dna_n,
        ],
        axis=1,
    )


def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def average_precision(ranked_relevances: np.ndarray) -> float:
    if ranked_relevances.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(ranked_relevances, start=1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions))


def compute_metrics(features: np.ndarray, labels: List[str], top_k: int) -> dict:
    labels = np.asarray(labels)
    feats = l2_normalize(features)
    sim = np.dot(feats, feats.T)

    n = len(labels)
    ap_scores = []
    precisions = []
    recalls = []
    f1s = []

    for i in range(n):
        # Exclude self
        scores = sim[i].copy()
        scores[i] = -np.inf

        ranked_idx = np.argsort(scores)[::-1]
        ranked_labels = labels[ranked_idx]
        relevances = (ranked_labels == labels[i]).astype(np.int32)

        # mAP
        ap_scores.append(average_precision(relevances))

        # Precision/Recall/F1 at K
        k = min(top_k, n - 1)
        topk_rel = relevances[:k]
        hits = int(topk_rel.sum())
        total_relevant = int((labels == labels[i]).sum() - 1)

        precision = hits / k if k > 0 else 0.0
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "f1@k": float(np.mean(f1s)),
        "mAP": float(np.mean(ap_scores)),
    }


def compute_precision_recall_vs_k(
    features: np.ndarray,
    labels: List[str],
    max_k: int,
) -> Tuple[List[int], List[float], List[float]]:
    labels = np.asarray(labels)
    feats = l2_normalize(features)
    sim = np.dot(feats, feats.T)
    n = len(labels)
    if n <= 1:
        return [], [], []

    if max_k <= 0:
        max_k = n - 1
    max_k = max(1, min(max_k, n - 1))
    ks = list(range(1, max_k + 1))
    precision_curve = []
    recall_curve = []

    for k in ks:
        precisions = []
        recalls = []
        for i in range(n):
            scores = sim[i].copy()
            scores[i] = -np.inf

            ranked_idx = np.argsort(scores)[::-1]
            ranked_labels = labels[ranked_idx]
            relevances = (ranked_labels == labels[i]).astype(np.int32)

            topk_rel = relevances[:k]
            hits = int(topk_rel.sum())
            total_relevant = int((labels == labels[i]).sum() - 1)

            precision = hits / k if k > 0 else 0.0
            recall = hits / total_relevant if total_relevant > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)

        precision_curve.append(float(np.mean(precisions)))
        recall_curve.append(float(np.mean(recalls)))

    return ks, precision_curve, recall_curve


def plot_precision_recall_vs_k(
    ks: List[int],
    precision_curve: List[float],
    recall_curve: List[float],
    output_path: str,
) -> None:
    if not ks:
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(ks, precision_curve, marker="o", linewidth=2, label="Precision@K")
    plt.plot(ks, recall_curve, marker="s", linewidth=2, label="Recall@K")
    plt.xlabel("K (number of retrieved items)")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs K")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_f1_vs_k(
    ks: List[int],
    precision_curve: List[float],
    recall_curve: List[float],
    output_path: str,
) -> None:
    if not ks:
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    p = np.asarray(precision_curve, dtype=np.float32)
    r = np.asarray(recall_curve, dtype=np.float32)
    f1 = (2.0 * p * r) / np.maximum(p + r, 1e-10)

    plt.figure(figsize=(9, 5))
    plt.plot(ks, f1.tolist(), marker="^", linewidth=2, label="F1@K")
    plt.xlabel("K (number of retrieved items)")
    plt.ylabel("F1")
    plt.title("F1 vs K")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_precision_vs_recall(
    recall_curve: List[float],
    precision_curve: List[float],
    output_path: str,
) -> None:
    if not recall_curve or not precision_curve:
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(recall_curve, precision_curve, marker="o", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_weight_search_history(
    scores: List[float],
    best_scores: List[float],
    output_path: str,
) -> None:
    if not scores:
        return

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    iters = list(range(1, len(scores) + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(iters, scores, linewidth=1.5, alpha=0.7, label="Iteration score")
    plt.plot(iters, best_scores, linewidth=2.5, label="Best-so-far score")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Fusion Weight Search History")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def optimize_fusion_weights(
    f_cnn: np.ndarray,
    f_dtcwt: np.ndarray,
    f_dna: np.ndarray,
    labels: List[str],
    top_k: int,
    num_samples: int,
    seed: int,
    target_metric: str,
    fixed_weights: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float], dict, dict]:
    if fixed_weights is not None:
        fused = weighted_fuse_features(
            f_cnn, f_dtcwt, f_dna, fixed_weights[0], fixed_weights[1], fixed_weights[2]
        )
        metrics = compute_metrics(fused, labels, top_k=top_k)
        history = {
            "scores": [float(metrics[target_metric])],
            "best_scores": [float(metrics[target_metric])],
            "weights": [tuple(np.asarray(fixed_weights, dtype=np.float32) / np.sum(fixed_weights))],
        }
        return fused, history["weights"][0], metrics, history

    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    best_metrics: Optional[dict] = None
    best_fused: Optional[np.ndarray] = None
    scores: List[float] = []
    best_scores: List[float] = []
    weight_list: List[Tuple[float, float, float]] = []

    for _ in range(num_samples):
        w = rng.dirichlet(np.ones(3, dtype=np.float32))
        weights = (float(w[0]), float(w[1]), float(w[2]))
        fused = weighted_fuse_features(f_cnn, f_dtcwt, f_dna, weights[0], weights[1], weights[2])
        metrics = compute_metrics(fused, labels, top_k=top_k)
        score = float(metrics[target_metric])
        scores.append(score)
        weight_list.append(weights)

        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics
            best_fused = fused
        best_scores.append(best_score)

    if best_metrics is None or best_fused is None:
        raise RuntimeError("Fusion weight search failed to produce a valid candidate.")

    history = {"scores": scores, "best_scores": best_scores, "weights": weight_list}
    return best_fused, best_weights, best_metrics, history


def main():
    parser = argparse.ArgumentParser(description="KTH retrieval evaluation with cosine similarity")
    parser.add_argument("--frames-dir", default="Frames", help="Frame root with class/video/I or P folders")
    parser.add_argument("--input-size", default=224, type=int, help="Input image size")
    parser.add_argument("--frames-per-video", default=10, type=int, help="Sampled frames per video")
    parser.add_argument("--batch-size", default=4, type=int, help="Batch size for feature extraction")
    parser.add_argument("--feature-dim", default=512, type=int, help="Feature dimension in head")
    parser.add_argument("--top-k", default=5, type=int, help="Top-K for precision/recall/F1")
    parser.add_argument(
        "--person-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use YOLO and extract features only from frames containing a human",
    )
    parser.add_argument(
        "--yolo-model-path",
        default="runs/detect/train2/weights/best.pt",
        help="Path to YOLO model weights used for person filtering",
    )
    parser.add_argument("--yolo-conf", default=0.25, type=float, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", default=0.5, type=float, help="YOLO IoU threshold")
    parser.add_argument("--yolo-batch-size", default=32, type=int, help="YOLO inference batch size")
    parser.add_argument(
        "--use-dtcwt-dna",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract DTCWT and DNA handcrafted features and fuse with CNN features",
    )
    parser.add_argument("--w-cnn", default=0.34, type=float, help="Fusion weight for CNN features")
    parser.add_argument("--w-dtcwt", default=0.33, type=float, help="Fusion weight for DTCWT features")
    parser.add_argument("--w-dna", default=0.33, type=float, help="Fusion weight for DNA features")
    parser.add_argument(
        "--optimize-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search multiple fusion weight combinations and report the best iteration",
    )
    parser.add_argument("--fusion-iters", default=200, type=int, help="Number of random fusion weight samples")
    parser.add_argument(
        "--fusion-target",
        default="mAP",
        choices=["mAP", "f1@k", "precision@k", "recall@k"],
        help="Metric to maximize when selecting best fusion weights",
    )
    parser.add_argument("--fusion-seed", default=42, type=int, help="Random seed for fusion weight search")
    parser.add_argument(
        "--max-k-curve",
        default=0,
        type=int,
        help="Maximum K for precision/recall curve; use 0 for full range (N-1)",
    )
    parser.add_argument(
        "--curve-path",
        default="Features/precision_recall_vs_k.png",
        help="Output path for precision/recall-vs-K plot",
    )
    parser.add_argument(
        "--pr-curve-path",
        default="Features/precision_vs_recall.png",
        help="Output path for precision-vs-recall plot",
    )
    parser.add_argument(
        "--f1-curve-path",
        default="Features/f1_vs_k.png",
        help="Output path for F1-vs-K plot",
    )
    parser.add_argument(
        "--fusion-history-path",
        default="Features/fusion_search_history.png",
        help="Output path for fusion weight search history plot",
    )
    parser.add_argument("--cache-dir", default="Features", help="Cache directory for features")
    parser.add_argument("--force", action="store_true", help="Recompute features even if cache exists")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_tag = f"{'person' if args.person_only else 'all'}_{MODEL_TAG}_{PREPROCESS_TAG}"
    feat_path = os.path.join(args.cache_dir, f"kth_features_{cache_tag}.npy")
    dtcwt_path = os.path.join(args.cache_dir, f"kth_dtcwt_{cache_tag}.npy")
    dna_path = os.path.join(args.cache_dir, f"kth_dna_{cache_tag}.npy")
    label_path = os.path.join(args.cache_dir, f"kth_labels_{cache_tag}.npy")
    path_path = os.path.join(args.cache_dir, f"kth_paths_{cache_tag}.npy")

    # Default to recompute on IDE runs to avoid stale features when changing architectures.
    force_recompute = True if not hasattr(args, "force") else bool(args.force)
    cache_ok = (not force_recompute) and os.path.exists(feat_path) and os.path.exists(label_path) and os.path.exists(path_path)
    if cache_ok:
        features = np.load(feat_path)
        labels = np.load(label_path).tolist()
        video_dirs = np.load(path_path).tolist()
        f_dtcwt = np.load(dtcwt_path) if (args.use_dtcwt_dna and os.path.exists(dtcwt_path)) else None
        f_dna = np.load(dna_path) if (args.use_dtcwt_dna and os.path.exists(dna_path)) else None
    else:
        if not os.path.isdir(args.frames_dir):
            raise FileNotFoundError(
                f"Frames directory '{args.frames_dir}' not found. Run Fram_Extraction.py first."
            )

        video_dirs, labels = list_videos_from_frames(args.frames_dir)
        if not video_dirs:
            raise RuntimeError("No frame folders found under Frames/<class>/<video>/{I|P}")

        model = build_feature_extractor(
            input_shape=(args.input_size, args.input_size, 3),
            feature_dim=args.feature_dim,
        )
        yolo_model = None
        if args.person_only:
            from ultralytics import YOLO #type: ignore

            if not os.path.exists(args.yolo_model_path):
                raise FileNotFoundError(
                    f"YOLO model not found at '{args.yolo_model_path}'. "
                    "Pass --yolo-model-path to the correct weights."
                )
            yolo_model = YOLO(args.yolo_model_path)

        features = extract_video_features(
            model,
            video_dirs,
            frames_per_video=args.frames_per_video,
            img_size=(args.input_size, args.input_size),
            batch_size=args.batch_size,
            yolo_model=yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_batch_size=args.yolo_batch_size,
        )
        f_dtcwt = None
        f_dna = None
        if args.use_dtcwt_dna:
            f_dtcwt, f_dna = extract_handcrafted_video_features(
                video_dirs=video_dirs,
                frames_per_video=args.frames_per_video,
                yolo_model=yolo_model,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_batch_size=args.yolo_batch_size,
            )
            np.save(dtcwt_path, f_dtcwt)
            np.save(dna_path, f_dna)

        np.save(feat_path, features)
        np.save(label_path, np.asarray(labels))
        np.save(path_path, np.asarray(video_dirs))

    fusion_history = None
    selected_weights = None
    if args.use_dtcwt_dna:
        if f_dtcwt is None or f_dna is None:
            # Cache had CNN features but not handcrafted ones; compute now.
            yolo_model = None
            if args.person_only:
                from ultralytics import YOLO #type: ignore

                if not os.path.exists(args.yolo_model_path):
                    raise FileNotFoundError(
                        f"YOLO model not found at '{args.yolo_model_path}'. "
                        "Pass --yolo-model-path to the correct weights."
                    )
                yolo_model = YOLO(args.yolo_model_path)
            f_dtcwt, f_dna = extract_handcrafted_video_features(
                video_dirs=video_dirs,
                frames_per_video=args.frames_per_video,
                yolo_model=yolo_model,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_batch_size=args.yolo_batch_size,
            )
            np.save(dtcwt_path, f_dtcwt)
            np.save(dna_path, f_dna)

        if args.optimize_fusion:
            retrieval_features, selected_weights, metrics, fusion_history = optimize_fusion_weights(
                f_cnn=features,
                f_dtcwt=f_dtcwt,
                f_dna=f_dna,
                labels=labels,
                top_k=args.top_k,
                num_samples=max(1, args.fusion_iters),
                seed=args.fusion_seed,
                target_metric=args.fusion_target,
                fixed_weights=None,
            )
        else:
            retrieval_features, selected_weights, metrics, fusion_history = optimize_fusion_weights(
                f_cnn=features,
                f_dtcwt=f_dtcwt,
                f_dna=f_dna,
                labels=labels,
                top_k=args.top_k,
                num_samples=1,
                seed=args.fusion_seed,
                target_metric=args.fusion_target,
                fixed_weights=(args.w_cnn, args.w_dtcwt, args.w_dna),
            )
    else:
        retrieval_features = features
        metrics = compute_metrics(retrieval_features, labels, top_k=args.top_k)
    ks, precision_curve, recall_curve = compute_precision_recall_vs_k(
        retrieval_features, labels, max_k=args.max_k_curve
    )
    plot_precision_recall_vs_k(ks, precision_curve, recall_curve, args.curve_path)
    plot_precision_vs_recall(recall_curve, precision_curve, args.pr_curve_path)
    plot_f1_vs_k(ks, precision_curve, recall_curve, args.f1_curve_path)
    if fusion_history is not None:
        plot_weight_search_history(
            fusion_history["scores"], fusion_history["best_scores"], args.fusion_history_path
        )

    print("KTH Retrieval Metrics")
    print(f"Videos: {len(labels)}")
    print(f"Precision@{args.top_k}: {metrics['precision@k']:.4f}")
    print(f"Recall@{args.top_k}: {metrics['recall@k']:.4f}")
    print(f"F1@{args.top_k}: {metrics['f1@k']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    if args.use_dtcwt_dna:
        if selected_weights is not None:
            w = np.asarray(selected_weights, dtype=np.float32)
            w = w / np.sum(w)
            label = "Best fusion weights" if args.optimize_fusion else "Fusion weights used"
            print(f"{label} (normalized): CNN={w[0]:.4f}, DTCWT={w[1]:.4f}, DNA={w[2]:.4f}")
            if args.optimize_fusion:
                print(f"Best iteration selected by: {args.fusion_target}")
    print(f"Precision/Recall vs K plot saved to: {args.curve_path}")
    print(f"Precision vs Recall plot saved to: {args.pr_curve_path}")
    print(f"F1 vs K plot saved to: {args.f1_curve_path}")
    if fusion_history is not None:
        print(f"Fusion search history plot saved to: {args.fusion_history_path}")

    # Log run details to Database folder for experiment tracking.
    os.makedirs("Database", exist_ok=True)
    log_path = os.path.join("Database", "cbir_run_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"Timestamp: {np.datetime64('now')}\n")
        f.write(f"Model tag: {MODEL_TAG}\n")
        f.write(f"Preprocess: {PREPROCESS_TAG}\n")
        f.write(f"Input size: {args.input_size}\n")
        f.write(f"Frames per video: {args.frames_per_video}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Person only: {args.person_only}\n")
        f.write(f"Use DTCWT/DNA: {args.use_dtcwt_dna}\n")
        f.write(f"Optimize fusion: {args.optimize_fusion}\n")
        f.write(f"Fusion target: {args.fusion_target}\n")
        f.write(f"Cache tag: {cache_tag}\n")
        f.write(f"Force recompute: {force_recompute}\n")
        f.write(f"Precision@{args.top_k}: {metrics['precision@k']:.4f}\n")
        f.write(f"Recall@{args.top_k}: {metrics['recall@k']:.4f}\n")
        f.write(f"F1@{args.top_k}: {metrics['f1@k']:.4f}\n")
        f.write(f"mAP: {metrics['mAP']:.4f}\n")
        if args.use_dtcwt_dna and selected_weights is not None:
            w = np.asarray(selected_weights, dtype=np.float32)
            w = w / np.sum(w)
            f.write(f"Fusion weights (normalized): CNN={w[0]:.4f}, DTCWT={w[1]:.4f}, DNA={w[2]:.4f}\n")
        f.write("\n")


if __name__ == "__main__":
    main()
