# Changelog

所有重要的项目变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added
- 新增 LGE 序列的 Low Signal 相关字段支持：
  - `Low Signal Abnormal Region` (LGE_sax, 多选题)：识别心肌低信号异常区域
  - `Low Signal Distribution Pattern` (LGE_sax, 多选题)：识别低信号分布模式
- 新增 Perfusion 序列的 `Abnormal Segments` 字段支持（perfusion, 多选题）：识别沿长轴的异常灌注节段
- 为所有新增字段添加了对应的 Reason Templates，提供5步结构化分析框架

### Changed
- 优化了 Low Signal Abnormal Region 的 prompt template：
  - 明确说明是 "myocardial low-signal abnormality (hypoenhanced / darker-than-expected area)"
  - 添加操作标准：需同时满足两个条件（比相邻心肌更暗 + 跨帧一致性）
  - 强调只标记心肌内的发现，排除 LV 腔/血池的暗区
- 优化了 Low Signal Distribution Pattern 的 prompt template：
  - 明确说明是 "myocardial low-signal abnormality"
  - 强调将模式标签应用于低信号区域本身
  - 操作性视觉定义更聚焦于低信号特征
- 优化了 Abnormal Segments (Perfusion) 的 prompt template：
  - 明确说明是 "slice-level segments along the long axis"
  - 强调评估跨时间帧的异常性（"wash-in as shown"）
- 更新了所有新增字段的 Reason Templates，使其更加详细和具体

### Fixed
- 修复了 LGE 序列和 Perfusion 序列中缺失的 prompt templates，确保所有数据集中的字段都有对应的 prompt 支持

## [Previous Versions]

### v2.0 - Prompt Templates 完整实现
- 实现了所有序列类型（Cine、LGE、Perfusion、T2）的专门 prompt templates
- 实现了结构化 Reason Templates（5步分析框架）
- 实现了 v2 设计标准：两行输出格式、Z. None 选项、图像仅推理

### v1.0 - 初始版本
- 基础评估框架
- 支持 GPT、Qwen、Ksyun 模型
- 支持序列过滤和批量测试
