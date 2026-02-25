import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img #type: ignore

from Effi_Res_hybrid import build_feature_extractor


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


def sample_frames(frame_dir: str, frames_per_video: int) -> List[str]:
    frames = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frames:
        return []

    if len(frames) <= frames_per_video:
        return [os.path.join(frame_dir, f) for f in frames]

    idxs = np.linspace(0, len(frames) - 1, frames_per_video, dtype=int)
    return [os.path.join(frame_dir, frames[i]) for i in idxs]


def load_batch(frame_paths: List[str], img_size: Tuple[int, int]) -> np.ndarray:
    batch = []
    for p in frame_paths:
        img = load_img(p, target_size=img_size)
        arr = img_to_array(img)
        batch.append(arr)
    batch = np.asarray(batch, dtype=np.float32)
    batch = batch / 255.0
    return batch


def extract_video_features(
    model: tf.keras.Model,
    video_dirs: List[str],
    frames_per_video: int,
    img_size: Tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    features = []

    for vdir in video_dirs:
        frame_paths = sample_frames(vdir, frames_per_video)
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


def main():
    parser = argparse.ArgumentParser(description="KTH retrieval evaluation with cosine similarity")
    parser.add_argument("--frames-dir", default="Frames", help="Frame root with class/video/I or P folders")
    parser.add_argument("--input-size", default=224, type=int, help="Input image size")
    parser.add_argument("--frames-per-video", default=10, type=int, help="Sampled frames per video")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size for feature extraction")
    parser.add_argument("--feature-dim", default=512, type=int, help="Feature dimension in head")
    parser.add_argument("--top-k", default=5, type=int, help="Top-K for precision/recall/F1")
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
    parser.add_argument("--cache-dir", default="Features", help="Cache directory for features")
    parser.add_argument("--force", action="store_true", help="Recompute features even if cache exists")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    feat_path = os.path.join(args.cache_dir, "kth_features.npy")
    label_path = os.path.join(args.cache_dir, "kth_labels.npy")
    path_path = os.path.join(args.cache_dir, "kth_paths.npy")

    if (not args.force) and os.path.exists(feat_path) and os.path.exists(label_path) and os.path.exists(path_path):
        features = np.load(feat_path)
        labels = np.load(label_path).tolist()
        video_dirs = np.load(path_path).tolist()
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

        features = extract_video_features(
            model,
            video_dirs,
            frames_per_video=args.frames_per_video,
            img_size=(args.input_size, args.input_size),
            batch_size=args.batch_size,
        )

        np.save(feat_path, features)
        np.save(label_path, np.asarray(labels))
        np.save(path_path, np.asarray(video_dirs))

    metrics = compute_metrics(features, labels, top_k=args.top_k)
    ks, precision_curve, recall_curve = compute_precision_recall_vs_k(
        features, labels, max_k=args.max_k_curve
    )
    plot_precision_recall_vs_k(ks, precision_curve, recall_curve, args.curve_path)
    plot_precision_vs_recall(recall_curve, precision_curve, args.pr_curve_path)

    print("KTH Retrieval Metrics")
    print(f"Videos: {len(labels)}")
    print(f"Precision@{args.top_k}: {metrics['precision@k']:.4f}")
    print(f"Recall@{args.top_k}: {metrics['recall@k']:.4f}")
    print(f"F1@{args.top_k}: {metrics['f1@k']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"Precision/Recall vs K plot saved to: {args.curve_path}")
    print(f"Precision vs Recall plot saved to: {args.pr_curve_path}")


if __name__ == "__main__":
    main()
