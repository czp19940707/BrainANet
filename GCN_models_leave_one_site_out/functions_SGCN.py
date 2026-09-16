import os
import re
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(
        labels,
        num_samples=None,
        replacement=True,
        seed=42
):
    """
    根据类别频数创建类别平衡的 WeightedRandomSampler。

    Parameters
    ----------
    labels : array-like
        训练集标签，例如 [0, 0, 0, 1, 1]。
        支持 list、NumPy array 或 Tensor。

    num_samples : int or None
        每个 epoch 抽取的样本总数。
        None 表示与原训练集样本数相同。

    replacement : bool
        是否采用有放回采样。
        类别不平衡任务通常设置为 True。

    seed : int
        随机种子。

    Returns
    -------
    sampler : WeightedRandomSampler
        可直接传入 DataLoader。

    sample_weights : torch.Tensor
        每个样本对应的采样权重。

    class_info : dict
        各类别的样本数量和类别权重。
    """
    labels = torch.as_tensor(labels).view(-1).long().cpu()

    if labels.numel() == 0:
        raise ValueError("labels不能为空。")

    # inverse_indices表示每个样本属于classes中的第几个类别
    classes, inverse_indices, class_counts = torch.unique(
        labels,
        sorted=True,
        return_inverse=True,
        return_counts=True
    )

    # 类别权重：样本数越少，权重越大
    class_weights = 1.0 / class_counts.float()

    # 将类别权重映射到每个样本
    sample_weights = class_weights[inverse_indices].double()

    if num_samples is None:
        num_samples = len(labels)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator
    )

    class_info = {
        int(class_label): {
            "count": int(count),
            "weight": float(weight)
        }
        for class_label, count, weight in zip(
            classes,
            class_counts,
            class_weights
        )
    }

    return sampler, sample_weights, class_info


# ============================================================
# 1. 随机种子
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. 处理模型输出
# ============================================================
def get_logits(model_output: torch.Tensor) -> torch.Tensor:
    """
    兼容以下模型输出形式：
    1. logits
    2. (logits, features)
    3. {"logits": logits}
    """
    if torch.is_tensor(model_output):
        return model_output

    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    if isinstance(model_output, dict):
        if "logits" not in model_output:
            raise KeyError("模型输出字典中不存在 'logits'。")
        return model_output["logits"]

    raise TypeError(
        f"无法识别模型输出类型：{type(model_output)}"
    )


# ============================================================
# 3. 单轮训练
# ============================================================
def train_one_epoch(
        model,
        data_loader,
        optimizer,
        device,
        class_weights=None,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        # images = batch["image"].to(
        #     device,
        #     non_blocking=True,
        # )
        # labels = batch["label"].to(
        #     device,
        #     non_blocking=True,
        # )
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        model_output = model(batch, False)
        logits = get_logits(model_output)

        # logits -> log probability
        loss_ce = F.log_softmax(logits, dim=1)

        model_output_prob = model(batch, True)
        logits_rpob = get_logits(model_output_prob)
        loss_mi = F.log_softmax(logits_rpob, dim=1)
        loss_prob = model.loss_probability(batch.x, batch.edge_index, batch.edge_attr)

        loss = loss_ce + 2 * loss_prob + loss_mi
        # labels = batch.y
        loss = F.nll_loss(
            loss,
            batch.y,
            weight=class_weights,
            reduction="mean",
        )

        loss.backward()
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


# ============================================================
# 4. 验证集评估
# ============================================================
@torch.no_grad()
def evaluate_model(
        model,
        data_loader,
        device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_prob_1 = []

    for batch in data_loader:
        # images = batch["image"].to(
        #     device,
        #     non_blocking=True,
        # )
        # labels = batch["label"].to(
        #     device,
        #     non_blocking=True,
        # )
        batch = batch.to(device)
        model_output = model(batch)
        logits = get_logits(model_output)

        log_probs = F.log_softmax(logits, dim=1)
        probabilities = torch.exp(log_probs)

        loss = F.nll_loss(
            log_probs,
            batch.y,
            reduction="mean",
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(
            batch.y.detach().cpu().numpy().tolist()
        )
        all_preds.extend(
            predictions.detach().cpu().numpy().tolist()
        )
        all_prob_1.extend(
            probabilities[:, 1]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    mean_loss = total_loss / max(total_samples, 1)

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    # 当验证集只有一个类别时，AUC 无法计算
    try:
        auc = roc_auc_score(
            all_labels,
            all_prob_1,
        )
    except ValueError:
        auc = np.nan

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "auc": auc,
    }


# ============================================================
# 5. 测试并保存每个样本的预测结果
# ============================================================
@torch.no_grad()
def predict_test_site(
        model,
        data_loader,
        device,
        test_site,
):
    model.eval()

    result_rows = []

    for batch in data_loader:
        # images = batch["image"].to(
        #     device,
        #     non_blocking=True,
        # )
        # labels = batch["label"].to(
        #     device,
        #     non_blocking=True,
        # )
        batch = batch.to(device)
        model_output = model(batch)
        logits = get_logits(model_output)

        # log_softmax 后使用 NLLLoss
        log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        # 二分类概率
        probabilities = torch.exp(log_probs)

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        # 每个样本对应一个 loss，而不是 batch 平均 loss
        nll_loss_no_reduction = F.nll_loss(
            log_probs,
            batch.y,
            reduction="none",
        )

        labels_np = batch.y.detach().cpu().numpy()
        predictions_np = (
            predictions.detach().cpu().numpy()
        )
        probabilities_np = (
            probabilities.detach().cpu().numpy()
        )
        losses_np = (
            nll_loss_no_reduction
            .detach()
            .cpu()
            .numpy()
        )

        subject_ids = list(batch.subject_id)
        phases = list(batch.site)
        groups = (batch.y.detach().cpu().numpy())

        for index in range(len(subject_ids)):
            result_rows.append(
                {
                    "TestSite": str(test_site),
                    "Subject ID": str(
                        subject_ids[index]
                    ),
                    "Phase": str(phases[index]),
                    "Group": int(groups[index]),
                    "Pred": int(
                        predictions_np[index]
                    ),
                    "Prob_0": float(
                        probabilities_np[index, 0]
                    ),
                    "Prob_1": float(
                        probabilities_np[index, 1]
                    ),
                    "loss": float(
                        losses_np[index]
                    ),
                }
            )

    return pd.DataFrame(result_rows)


# ============================================================
# 6. 计算测试中心指标
# ============================================================
def calculate_binary_metrics(
        prediction_dataframe,
        # group_to_label,
):
    """
    group_to_label 示例：
    {
        "NC": 0,
        "AD": 1
    }
    """
    true_labels = (
        prediction_dataframe["Group"]
        # .map(group_to_label)
        .to_numpy()
    )

    predictions = (
        prediction_dataframe["Pred"]
        .astype(int)
        .to_numpy()
    )

    prob_1 = (
        prediction_dataframe["Prob_1"]
        .astype(float)
        .to_numpy()
    )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    try:
        auc = roc_auc_score(
            true_labels,
            prob_1,
        )
    except ValueError:
        auc = np.nan

    confusion = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = confusion.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "ACC": accuracy,
        "SEN": sensitivity,
        "SPE": specificity,
        "AUC": auc,
        "MeanLoss": prediction_dataframe[
            "loss"
        ].mean(),
        "N": len(prediction_dataframe),
    }


# ============================================================
# 8. Leave-one-site 主函数
# ============================================================
def run_leave_one_site_cv(
        metadata,
        dataset,
        model_factory,
        output_dir,
        n_node=90,
        net_name="R2SN",
        model_name="GCN",
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        val_ratio=None,
        num_workers=0,
        patience=20,
        seed=42,
        task='ASD',
):
    """
    保留 leave-one-site 训练模式，但不再划分验证集。
    val_ratio 参数仅为兼容旧调用保留，不再参与数据划分。

    每个 fold：
        1. 当前 site 作为独立测试中心；
        2. 其余所有 site 作为训练集；
        3. 每个 epoch 结束后在测试中心上评估；
        4. 测试 AUC 最高时保存模型；
        5. 若 AUC 相同，则优先保留测试 loss 更低的模型；
        6. 若测试中心只有一个类别、AUC 无法计算，则退化为按测试 loss 选择。

    注意
    ----
    使用测试中心 AUC 选择 epoch 会使测试集参与模型选择，得到的测试性能
    将存在乐观偏差。该流程适合模型调试或探索性实验，不宜作为严格独立测试结果。

    metadata 必须和 dataset 顺序完全一致，并至少包含：
        Stage
        Label

    dataset 中每个 Data 对象应包含：
        y
        subject_id
        site

    model_factory 示例：
        model_factory=lambda net, model, n_node: MyModel(...)
    """
    seed_everything(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sites = sorted(
        metadata["Stage"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    all_site_predictions = []
    all_site_metrics = []

    print(f"共检测到 {len(sites)} 个 site：")
    print(sites)

    for fold_index, test_site in enumerate(sites, start=1):
        print("\n" + "=" * 80)
        print(
            f"Fold {fold_index}/{len(sites)} | "
            f"Test site: {test_site}"
        )
        print("=" * 80)

        fold_seed = seed + fold_index
        seed_everything(fold_seed)

        site_series = metadata["Stage"].astype(str)

        test_indices = metadata.index[
            site_series == str(test_site)
        ].to_numpy()

        # 取消验证集：除测试中心之外的全部样本用于训练
        train_indices = metadata.index[
            site_series != str(test_site)
        ].to_numpy()

        if len(test_indices) == 0:
            print(f"Site {test_site} 没有测试样本，跳过。")
            continue

        if len(train_indices) == 0:
            print(f"Site {test_site} 之外没有训练样本，跳过。")
            continue

        train_label_values = metadata.loc[
            train_indices, "Label"
        ].astype(int).to_numpy()
        test_label_values = metadata.loc[
            test_indices, "Label"
        ].astype(int).to_numpy()

        print(
            f"Train: {len(train_indices)}, "
            f"Test: {len(test_indices)}"
        )
        print(
            "Train label distribution: "
            f"{dict(zip(*np.unique(train_label_values, return_counts=True)))}"
        )
        print(
            "Test label distribution: "
            f"{dict(zip(*np.unique(test_label_values, return_counts=True)))}"
        )

        test_has_two_classes = np.unique(test_label_values).size == 2
        if not test_has_two_classes:
            print(
                f"警告：测试中心 {test_site} 只包含一个类别，"
                "AUC 无法计算，将使用测试 loss 选择最佳 epoch。"
            )

        train_dataset = Subset(
            dataset,
            train_indices.tolist(),
        )
        test_dataset = Subset(
            dataset,
            test_indices.tolist(),
        )

        # 仅在训练集上进行类别平衡采样
        train_labels = [
            int(data.y.view(-1)[0].item())
            for data in train_dataset
        ]

        sampler, sample_weights, class_info = create_balanced_sampler(
            labels=train_labels,
            num_samples=len(train_dataset),
            replacement=True,
            seed=fold_seed,
        )

        print("Balanced sampler class info:", class_info)

        pin_memory = device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # 每个 fold 初始化全新模型
        model = model_factory(
            net=net_name,
            model=model_name,
            n_node=n_node,
            task=task,

        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 按测试 AUC 调整学习率；若 AUC 为 NaN，则传入 -TestLoss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        site_name = str(test_site)

        fold_dir = (
            output_dir /
            f"fold_{fold_index:02d}_test_{site_name}"
        )
        fold_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = fold_dir / "best_model.pth"
        history_path = fold_dir / "training_history.csv"
        site_prediction_path = (
            fold_dir / f"predictions_{site_name}.csv"
        )

        best_test_auc = -np.inf
        best_test_loss = float("inf")
        best_epoch = 0
        early_stop_counter = 0
        training_history = []

        # ----------------------------------------------------
        # 模型训练：每个 epoch 在测试中心评估
        # ----------------------------------------------------
        for epoch in range(1, num_epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

            current_prediction_dataframe = predict_test_site(
                model=model,
                data_loader=test_loader,
                device=device,
                test_site=test_site,
            )

            current_metrics = calculate_binary_metrics(
                prediction_dataframe=current_prediction_dataframe,
            )

            current_test_auc = current_metrics["AUC"]
            current_test_loss = current_metrics["MeanLoss"]

            # AUC 可计算时按 AUC 最大化；AUC 相同时按 loss 最小化
            if np.isfinite(current_test_auc):
                auc_improved = current_test_auc > best_test_auc
                auc_tied = (
                    np.isfinite(best_test_auc)
                    and np.isclose(
                        current_test_auc,
                        best_test_auc,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                is_best = auc_improved or (
                    auc_tied
                    and current_test_loss < best_test_loss
                )
                scheduler_score = current_test_auc
            else:
                # 测试中心只有一个类别时，无法使用 AUC
                is_best = (
                    not np.isfinite(best_test_auc)
                    and current_test_loss < best_test_loss
                )
                scheduler_score = -current_test_loss

            if is_best:
                best_test_auc = (
                    float(current_test_auc)
                    if np.isfinite(current_test_auc)
                    else np.nan
                )
                best_test_loss = float(current_test_loss)
                best_epoch = epoch
                early_stop_counter = 0

                torch.save(
                    {
                        "fold": fold_index,
                        "test_site": test_site,
                        "epoch": epoch,
                        "model_state_dict": copy.deepcopy(
                            model.state_dict()
                        ),
                        "optimizer_state_dict": copy.deepcopy(
                            optimizer.state_dict()
                        ),
                        "best_test_auc": best_test_auc,
                        "best_test_loss": best_test_loss,
                        "best_test_metrics": copy.deepcopy(
                            current_metrics
                        ),
                    },
                    checkpoint_path,
                )

                # 在 AUC 最佳时同步保留逐样本预测结果
                current_prediction_dataframe.to_csv(
                    site_prediction_path,
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                early_stop_counter += 1

            scheduler.step(scheduler_score)
            current_lr = optimizer.param_groups[0]["lr"]

            training_history.append(
                {
                    "Epoch": epoch,
                    "TrainLoss": train_loss,
                    "TestLoss": current_test_loss,
                    "TestACC": current_metrics["ACC"],
                    "TestSEN": current_metrics["SEN"],
                    "TestSPE": current_metrics["SPE"],
                    "TestAUC": current_test_auc,
                    "LearningRate": current_lr,
                    "IsBest": int(is_best),
                }
            )

            auc_text = (
                f"{current_test_auc:.4f}"
                if np.isfinite(current_test_auc)
                else "nan"
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss:.6f} | "
                f"Test loss: {current_test_loss:.6f} | "
                f"Test ACC: {current_metrics['ACC']:.4f} | "
                f"Test SEN: {current_metrics['SEN']:.4f} | "
                f"Test SPE: {current_metrics['SPE']:.4f} | "
                f"Test AUC: {auc_text} | "
                f"Best: {is_best}"
            )

            if early_stop_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch}, "
                    f"best epoch: {best_epoch}"
                )
                break

        pd.DataFrame(training_history).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Fold {fold_index} 未保存任何 checkpoint。"
            )

        # ----------------------------------------------------
        # 加载测试 AUC 最优模型
        # ----------------------------------------------------
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        checkpoint_auc = checkpoint["best_test_auc"]
        checkpoint_auc_text = (
            f"{checkpoint_auc:.6f}"
            if np.isfinite(checkpoint_auc)
            else "nan"
        )

        print(
            f"加载最佳模型：epoch {checkpoint['epoch']}, "
            f"test AUC = {checkpoint_auc_text}, "
            f"test loss = {checkpoint['best_test_loss']:.6f}"
        )

        # 重新由最佳 checkpoint 生成最终逐样本结果
        site_prediction_dataframe = predict_test_site(
            model=model,
            data_loader=test_loader,
            device=device,
            test_site=test_site,
        )

        site_prediction_dataframe.to_csv(
            site_prediction_path,
            index=False,
            encoding="utf-8-sig",
        )

        site_metrics = calculate_binary_metrics(
            prediction_dataframe=site_prediction_dataframe,
        )

        site_metrics.update(
            {
                "Fold": fold_index,
                "TestSite": test_site,
                "BestEpoch": best_epoch,
                "BestTestAUC": best_test_auc,
                "BestTestLoss": best_test_loss,
            }
        )

        all_site_metrics.append(site_metrics)
        all_site_predictions.append(
            site_prediction_dataframe
        )

        print(
            f"Best test site {test_site} | "
            f"ACC: {site_metrics['ACC']:.4f} | "
            f"SEN: {site_metrics['SEN']:.4f} | "
            f"SPE: {site_metrics['SPE']:.4f} | "
            f"AUC: {site_metrics['AUC']:.4f}"
        )

    # ========================================================
    # 合并所有测试 site 的样本结果
    # ========================================================
    if all_site_predictions:
        all_predictions_dataframe = pd.concat(
            all_site_predictions,
            axis=0,
            ignore_index=True,
        )

        all_predictions_dataframe.to_csv(
            output_dir / "all_test_sites_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        all_predictions_dataframe = pd.DataFrame()

    # ========================================================
    # 保存每个 site 的指标
    # ========================================================
    metrics_dataframe = pd.DataFrame(all_site_metrics)

    metrics_dataframe.to_csv(
        output_dir / "leave_one_site_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nLeave-one-site 训练完成。")
    print(f"结果保存在：{output_dir.resolve()}")

    return all_predictions_dataframe, metrics_dataframe
