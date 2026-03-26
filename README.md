# FPGA-based CNN Face Recognition System
**基於卷積神經網路之人臉辨識的 FPGA 硬體實現**

![FPGA](https://img.shields.io/badge/Platform-PYNQ--Z1-blue) 
![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green) 
![Tools](https://img.shields.io/badge/Tools-Vivado%20%7C%20Vitis%20HLS-orange)
![AI](https://img.shields.io/badge/Model-MobileFaceNet-red)

## 📖 專題簡介 (Project Overview)
本專案在資源受限的 **PYNQ-Z1 嵌入式硬體平台**上，結合卷積神經網路（CNN）與 FPGA 軟硬體協同設計（HW/SW Co-design），打造一套**低成本、低功耗且完全離線獨立運作**的 AI 臉部辨識門禁系統。

系統整合了 Haar 特徵級聯分類器進行輕量化人臉偵測，並使用專為邊緣裝置設計的輕量級模型 **MobileFaceNet** 進行特徵擷取。透過將底層運算轉換為硬體 IP（Hardware IP）部署於 FPGA 上，大幅加速了深度學習模型的推論效能。

## ✨ 系統亮點與效能優化 (Technical Highlights)
本系統不僅僅是模型的部署，更針對邊緣運算環境進行了多項底層架構優化：

*   🚀 **模型維度分析與取捨 (Model Dimensionality Trade-off)**：
    針對嵌入式設備記憶體受限的特性，實作並比較了 512維、256維與 128維特徵輸出模型。最終選定 **128維特徵模型**，其辨識速度最快（僅需 19.36 ms），且仍維持 78.97% 的準確率與 0.77 的 F1-score，達成效能與精準度的最佳平衡。
*   ⚡ **Vitis HLS 硬體加速指令優化 (Hardware Acceleration)**：
    針對主要運算層導入 C++ HLS 優化指令，突破效能瓶頸：
    *   `#pragma HLS PIPELINE II=1`：實現循環管線化，提升運算並行度。
    *   `#pragma HLS ARRAY_PARTITION`：將特徵與權重陣列分割，提升記憶體頻寬利用率。
    *   **結果**：成功將模型推論時間減少約 6.04%。
*   🧠 **AXI 介面與實體記憶體管理 (Memory Management)**：
    為解決 Python 虛擬記憶體與硬體實體記憶體間的轉換問題，透過 `allocate` 分配實體連續（physically contiguous）記憶體，並利用 AXI 介面讓 FPGA 直接存取 DRAM 資料，確保模型權重與特徵圖的高速傳輸。

## 🏗️ 系統架構 (System Architecture)
本系統採用 PS（Processing System）與 PL（Programmable Logic）協同運作的架構：
*   **PS 端 (ARM Cortex-A9)**：基於 Linux 作業系統與 Jupyter Notebook，負責攝影機影像擷取、系統控制邏輯（Driver控制）、實體按鈕事件處理與記憶體分配。
*   **PL 端 (FPGA 可程式邏輯)**：透過 Vitis HLS 將 Convolution, BatchNorm1D/2D, PReLU 等神經網路運算單元封裝為硬體 IP，處理密集的矩陣運算加速。

## 🛠️ 開發環境與硬體需求 (Requirements)
*   **硬體**：PYNQ-Z1 開發板、USB 攝影機。
*   **軟體/開發工具**：Xilinx Vivado 2022.1、Vitis HLS 2022.1、Jupyter Notebook、Python、OpenCV。

## 🚀 快速開始 (Quick Start / Demo)
*(建議在此處附上一張系統運作的 GIF 動畫或截圖)*

系統已具備完整的離線獨立運作能力，透過 PYNQ-Z1 板上的實體按鈕即可完成所有操作：
1.  **啟動與偵測**：連接 USB 攝影機，系統將自動啟動 Haar 模型捕捉畫面中的人臉。
2.  **拍攝擷取（Button 0）**：按下 `按鈕 0`，系統會擷取當前人臉並自動裁切、縮放至 112x112 解析度。
3.  **離線註冊（Button 1）**：按下 `按鈕 1` 並輸入使用者名稱，系統會透過硬體加速提取 128 維特徵向量並儲存於本地端（`.txt` 格式）。
4.  **即時辨識（Button 2）**：拍攝新照片後按下 `按鈕 2`，系統將提取新特徵，並與資料庫進行餘弦相似度（Cosine Similarity）比對。相似度大於閾值即顯示辨識成功。

## 👥 專題團隊 (Contributors)
*   國立東華大學資訊工程學系 (113學年度)
*   王澤鈞、陳昱丞、張芷榆、陳品秀
*   指導教授：紀新洲 教授
