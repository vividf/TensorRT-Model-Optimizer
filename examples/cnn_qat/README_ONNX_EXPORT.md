# QAT 模型 ONNX 導出指南

本指南說明如何使用修改後的 `torchvision_qat.py` 腳本來訓練 QAT 模型並自動導出為 ONNX 格式。

## 功能特點

- ✅ 自動進行 PTQ (Post-Training Quantization) 校準
- ✅ 執行 QAT (Quantization-Aware Training) 訓練
- ✅ 自動保存最佳模型檢查點
- ✅ **新增：自動導出 ONNX 格式模型**

## 使用方法

### 基本使用

```bash
python torchvision_qat.py \
    --train-data-path /workspace/data/train.X1/ \
    --val-data-path /workspace/data/val.X/ \
    --batch-size 16 \
    --num-workers 8 \
    --epochs 1 \
    --lr 1e-4 \
    --print-freq 50 \
    --output-dir ./resnet50_qat_output
```

### 參數說明

#### 必需參數
- `--train-data-path`: 訓練數據集路徑 (ImageNet 格式，每個類別一個子文件夾)
- `--val-data-path`: 驗證數據集路徑 (ImageNet 格式，每個類別一個子文件夾)

#### 可選參數
- `--batch-size`: 批次大小 (預設: 32)
- `--num-workers`: 數據加載器工作進程數 (預設: 4)
- `--epochs`: 訓練輪數 (預設: 5)
- `--lr`: 學習率 (預設: 0.001)
- `--momentum`: SGD 動量 (預設: 0.9)
- `--weight-decay`: 權重衰減 (預設: 1e-4)
- `--seed`: 隨機種子 (預設: 42)
- `--gpu`: GPU ID (單 GPU 訓練時使用)
- `--print-freq`: 打印頻率 (預設: 100)
- `--output-dir`: 輸出目錄 (預設: ".")

## 輸出文件

訓練完成後，在指定的輸出目錄中會生成以下文件：

1. **模型檢查點文件**:
   - `resnet50_qat_epoch_X.pth`: 每個 epoch 的最佳模型檢查點

2. **ONNX 模型文件**:
   - `resnet50_qat.onnx`: 標準 ONNX 格式模型
   - `resnet50_qat_int8.onnx`: 量化版本的 ONNX 模型

## 工作流程

1. **FP32 基準評估**: 評估原始 FP32 模型的準確率
2. **PTQ 校準**: 使用 512 張圖像進行校準
3. **QAT 訓練**: 執行量化感知訓練
4. **模型保存**: 保存最佳檢查點
5. **ONNX 導出**: 自動導出為 ONNX 格式

## 測試 ONNX 導出功能

在運行完整訓練之前，您可以先測試 ONNX 導出功能：

```bash
python test_onnx_export.py
```

這將創建一個測試 ONNX 文件來驗證導出功能是否正常工作。

## 多 GPU 訓練

使用 `torchrun` 進行多 GPU 訓練：

```bash
torchrun --nproc_per_node=2 torchvision_qat.py \
    --train-data-path /workspace/data/train.X1/ \
    --val-data-path /workspace/data/val.X/ \
    --batch-size 16 \
    --num-workers 8 \
    --epochs 1 \
    --lr 1e-4 \
    --print-freq 50 \
    --output-dir ./resnet50_qat_output
```

## 故障排除

### 常見問題

1. **CUDA 內存不足**:
   - 減少 `--batch-size`
   - 減少 `--num-workers`

2. **ONNX 導出失敗**:
   - 確保已安裝 `onnx` 包: `pip install onnx`
   - 檢查模型是否在正確的設備上 (CUDA)

3. **數據加載錯誤**:
   - 確保數據集路徑正確
   - 檢查數據集格式是否為 ImageNet 格式

### 日誌輸出示例

```
Evaluating FP32 baseline...
Original FP32 model accuracy: 76.13%
Starting PTQ...
PTQ model accuracy: 75.89%
Starting QAT for 1 epochs...
Epoch 1/1
QAT accuracy: 76.45%
Saved best QAT model: ./resnet50_qat_output/resnet50_qat_epoch_1.pth (Acc: 76.45%)
Training complete in 45.23s
Exporting QAT model to ONNX...
Successfully exported ONNX model to: ./resnet50_qat_output/resnet50_qat.onnx
```

## 依賴要求

確保安裝以下依賴：

```bash
pip install torch torchvision onnx modelopt
```

## 注意事項

- ONNX 導出僅在主進程 (rank 0) 上執行
- 導出的 ONNX 模型支持動態批次大小
- 量化模型會自動處理量化參數的導出
- 如果 ModelOpt 導出失敗，會自動嘗試標準 PyTorch ONNX 導出 