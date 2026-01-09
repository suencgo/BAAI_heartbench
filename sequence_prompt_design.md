# Sequence Prompt and Reason Template Design

本文档展示了为 LGE、T2 和 Perfusion 序列设计的 Prompt Templates 和 Reason Templates。

## 1. LGE (Late Gadolinium Enhancement) 序列

### 1.1 Enhancement Status (强化状态) - 单选题 - LGE_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, determine whether abnormal delayed enhancement is present (single-choice).

Strict rules:
- Choose only A or B; select exactly one.
- Output MUST be a single letter only (A or B).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, choose the most likely answer.

Options:
A. No Abnormal Delayed Enhancement
B. Abnormal Delayed Enhancement
```

**Reason Template:**
```
Your reason should include:
1. **Enhancement detection**: Look for areas of high signal intensity (bright areas) in the myocardium that indicate delayed gadolinium enhancement
2. **Normal vs abnormal comparison**: Compare the signal intensity of myocardial regions to normal myocardium
3. **Location identification**: Note where any abnormal enhancement is located (if present)
4. **Signal characteristics**: Describe the signal intensity pattern (uniform vs. patchy, etc.)
5. **Overall assessment**: Conclude whether abnormal delayed enhancement is present or absent

Example structure: "The myocardium shows [normal/abnormal] signal intensity. [Specific location observations]. [Signal pattern description]. This indicates [no abnormal delayed enhancement/abnormal delayed enhancement present]."
```

### 1.2 Abnormal Signal (异常信号) - 单选题 - LGE_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, determine the type of abnormal signal present (single-choice).

Strict rules:
- Choose only A or B; select exactly one.
- Output MUST be a single letter only (A or B).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, choose the most likely answer.

Options:
A. High Signal
B. Mixed Signal
```

**Reason Template:**
```
Your reason should include:
1. **Signal intensity analysis**: Assess whether abnormal areas show uniformly high signal or mixed signal patterns
2. **Signal distribution**: Describe how the signal is distributed (homogeneous high signal vs. mixed high and low signal areas)
3. **Contrast characteristics**: Compare signal intensity of abnormal areas to normal myocardium and blood pool
4. **Pattern identification**: Identify whether the pattern is predominantly high signal or shows mixed signal characteristics
5. **Overall classification**: Conclude whether the abnormal signal is high signal or mixed signal

Example structure: "The abnormal enhancement areas show [high/mixed] signal intensity. [Signal distribution description]. [Contrast comparison]. This indicates [high signal/mixed signal] pattern."
```

### 1.3 High Signal Abnormal Segment (高信号异常节段) - 多选题 - LGE_4ch

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, identify which segments show high signal abnormal enhancement (multi-select).

Strict rules:
- Select only from A/B/C/D; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Basal Segment
B. Mid Segment
C. Apical Segment
D. Apex
```

**Reason Template:**
```
Your reason should include:
1. **Segment identification**: Identify the cardiac segments (basal, mid, apical, apex) visible in the four-chamber view
2. **Enhancement location**: Determine which segments show high signal delayed enhancement
3. **Long-axis distribution**: Describe the distribution of enhancement along the long axis of the left ventricle
4. **Segment-by-segment analysis**: Systematically evaluate each segment for presence of high signal enhancement
5. **Overall pattern**: Summarize which segments are involved

Example structure: "High signal enhancement is observed in [specific segments]. [Long-axis distribution description]. [Segment-by-segment findings]. The involved segments are [list segments]."
```

### 1.4 High Signal Abnormal Region (高信号异常分区) - 多选题 - LGE_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, identify which wall regions show high signal abnormal enhancement (multi-select).

Strict rules:
- Select only from A/B/C/D/E/F; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Anterior Wall
B. Anteroseptal Wall
C. Inferoseptal Wall
D. Inferior Wall
E. Inferolateral Wall
F. Anterolateral Wall
```

**Reason Template:**
```
Your reason should include:
1. **Wall region identification**: Identify the six standard wall regions in the short-axis view
2. **Enhancement mapping**: Map the location of high signal enhancement to specific wall regions
3. **Circumferential distribution**: Describe the circumferential distribution of enhancement around the LV
4. **Regional analysis**: Systematically evaluate each wall region for presence of high signal enhancement
5. **Spatial pattern**: Describe the spatial pattern of involvement (e.g., anterior, inferior, lateral, septal)

Example structure: "High signal enhancement is observed in [specific wall regions]. [Circumferential distribution]. [Regional analysis findings]. The involved regions are [list regions]."
```

### 1.5 High Signal Distribution Pattern (高信号分布形状) - 多选题 - LGE_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, identify the distribution pattern of high signal enhancement (multi-select).

Strict rules:
- Select only from A/B/C/D/E; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Diffuse
B. Linear
C. Patchy
D. Transmural
E. Speckled
```

**Reason Template:**
```
Your reason should include:
1. **Spatial distribution analysis**: Assess how the high signal enhancement is distributed spatially across the myocardium
2. **Pattern characteristics**: Identify whether the pattern is diffuse (widespread), linear (line-like), patchy (irregular patches), transmural (full thickness), or speckled (small scattered areas)
3. **Geometric description**: Describe the geometric shape and extent of enhancement areas
4. **Continuity assessment**: Evaluate whether enhancement is continuous or discontinuous
5. **Overall pattern classification**: Classify the distribution pattern based on spatial characteristics

Example structure: "The high signal enhancement shows [pattern type] distribution. [Spatial characteristics]. [Geometric description]. [Continuity assessment]. This indicates [pattern classification]."
```

### 1.6 High Signal Myocardial Layer (高信号心肌层) - 多选题 - LGE_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, identify which myocardial layers show high signal enhancement (multi-select).

Strict rules:
- Select only from A/B/C/D/E; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Subendocardial
B. Mid-myocardial
C. Transmural
D. Subepicardial
E. Papillary Muscle
```

**Reason Template:**
```
Your reason should include:
1. **Myocardial layer identification**: Identify the different myocardial layers (subendocardial, mid-myocardial, subepicardial, transmural)
2. **Enhancement depth analysis**: Determine the depth of enhancement within the myocardial wall
3. **Layer-by-layer assessment**: Systematically evaluate each layer for presence of high signal enhancement
4. **Wall thickness involvement**: Assess whether enhancement involves partial or full wall thickness
5. **Papillary muscle evaluation**: If visible, assess papillary muscle involvement

Example structure: "High signal enhancement is observed in [specific layers]. [Depth analysis]. [Layer-by-layer findings]. [Wall thickness involvement]. [Papillary muscle status]. The involved layers are [list layers]."
```

### 1.7 Special Description (特殊描述) - 多选题 - LGE_4ch

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac LGE MRI. Task: using ONLY the provided image frames, identify any special features or abnormalities (multi-select).

Strict rules:
- Select only from A/B/C/D; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. None
B. Pericardial Abnormality
C. High Signal at Ventricular Septal Insertion Point
D. High Signal in Right Ventricle
```

**Reason Template:**
```
Your reason should include:
1. **Pericardium assessment**: Examine the pericardium for any abnormalities or enhancement
2. **Septal insertion point evaluation**: Look for high signal at the ventricular septal insertion points
3. **Right ventricle analysis**: Assess the right ventricle for presence of high signal enhancement
4. **Special feature identification**: Identify any unusual or characteristic findings beyond standard myocardial enhancement
5. **Overall special findings**: Summarize any special features present or confirm absence

Example structure: "Special findings include [specific features or none]. [Pericardium status]. [Septal insertion point status]. [Right ventricle status]. [Other special features]."
```

---

## 2. Perfusion (灌注) 序列

### 2.1 Perfusion Status (灌注状态) - 单选题 - perfusion

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac perfusion MRI. Task: using ONLY the provided image frames, determine the myocardial perfusion status (single-choice).

Strict rules:
- Choose only A or B; select exactly one.
- Output MUST be a single letter only (A or B).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, choose the most likely answer.

Options:
A. Normal
B. Abnormal
```

**Reason Template:**
```
Your reason should include:
1. **Perfusion signal assessment**: Evaluate the myocardial signal intensity during first-pass contrast enhancement
2. **Regional perfusion comparison**: Compare perfusion signal across different myocardial regions
3. **Contrast enhancement pattern**: Assess the pattern of contrast agent arrival and distribution
4. **Perfusion defects identification**: Look for areas of reduced or absent contrast enhancement (perfusion defects)
5. **Overall perfusion status**: Conclude whether perfusion appears normal or abnormal

Example structure: "The myocardial perfusion shows [normal/abnormal] pattern. [Signal intensity assessment]. [Regional comparison]. [Contrast enhancement pattern]. [Perfusion defects if present]. This indicates [normal/abnormal] perfusion status."
```

### 2.2 Abnormal Regions (异常区域) - 多选题 - perfusion

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac perfusion MRI. Task: using ONLY the provided image frames, identify which wall regions show abnormal perfusion (multi-select).

Strict rules:
- Select only from A/B/C/D/E/F; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Anterior wall
B. Anteroseptal wall
C. Inferoseptal wall
D. Inferior wall
E. Inferolateral wall
F. Anterolateral wall
```

**Reason Template:**
```
Your reason should include:
1. **Wall region identification**: Identify the six standard wall regions in the short-axis view
2. **Perfusion defect mapping**: Map the location of reduced or absent perfusion to specific wall regions
3. **Circumferential distribution**: Describe the circumferential distribution of perfusion abnormalities
4. **Regional perfusion analysis**: Systematically evaluate each wall region for perfusion defects
5. **Spatial pattern**: Describe the spatial pattern of perfusion abnormalities

Example structure: "Perfusion abnormalities are observed in [specific wall regions]. [Circumferential distribution]. [Regional analysis findings]. The involved regions are [list regions]."
```

### 2.3 Perfusion Abnormality Signal Characteristics (灌注异常信号特征) - 单选题 - perfusion

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac perfusion MRI. Task: using ONLY the provided image frames, determine the signal characteristics of perfusion abnormalities (single-choice).

Strict rules:
- Choose only A or B or C; select exactly one.
- Output MUST be a single letter only (A or B or C).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, choose the most likely answer.

Options:
A. Reduced perfusion
B. Delayed perfusion
C. Perfusion defect
```

**Reason Template:**
```
Your reason should include:
1. **Signal intensity analysis**: Assess the signal intensity in abnormal regions compared to normal myocardium
2. **Contrast arrival timing**: Evaluate the timing of contrast agent arrival in abnormal regions
3. **Perfusion defect characteristics**: Identify whether regions show reduced signal (reduced perfusion), delayed enhancement (delayed perfusion), or complete absence of enhancement (perfusion defect)
4. **Temporal pattern**: Assess the temporal evolution of contrast enhancement in abnormal regions
5. **Overall classification**: Classify the perfusion abnormality type based on signal characteristics

Example structure: "The perfusion abnormality shows [reduced/delayed/defect] characteristics. [Signal intensity comparison]. [Contrast arrival timing]. [Temporal pattern]. This indicates [reduced perfusion/delayed perfusion/perfusion defect]."
```

### 2.4 Myocardial Layer (心肌层) - 多选题 - perfusion

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac perfusion MRI. Task: using ONLY the provided image frames, identify which myocardial layers show perfusion abnormalities (multi-select).

Strict rules:
- Select only from A/B/C/D/E; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Subendocardial
B. Mid-myocardial
C. Subepicardial
D. Transmural
E. Papillary muscle
```

**Reason Template:**
```
Your reason should include:
1. **Myocardial layer identification**: Identify the different myocardial layers (subendocardial, mid-myocardial, subepicardial, transmural)
2. **Perfusion defect depth analysis**: Determine the depth of perfusion abnormalities within the myocardial wall
3. **Layer-by-layer assessment**: Systematically evaluate each layer for perfusion defects
4. **Wall thickness involvement**: Assess whether perfusion abnormalities involve partial or full wall thickness
5. **Papillary muscle evaluation**: If visible, assess papillary muscle perfusion

Example structure: "Perfusion abnormalities are observed in [specific layers]. [Depth analysis]. [Layer-by-layer findings]. [Wall thickness involvement]. [Papillary muscle status]. The involved layers are [list layers]."
```

---

## 3. T2 (T2-weighted) 序列

### 3.1 T2 Signal (T2信号) - 单选题 - T2_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI. Task: using ONLY the provided image frames, determine the T2 signal characteristics (single-choice).

Strict rules:
- Choose only A or B or C or D; select exactly one.
- Output MUST be a single letter only (A or B or C or D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, choose the most likely answer.

Options:
A. Normal
B. High Signal
C. Low Signal
D. Mixed signal
```

**Reason Template:**
```
Your reason should include:
1. **T2 signal intensity assessment**: Evaluate the T2-weighted signal intensity of the myocardium
2. **Normal myocardium comparison**: Compare myocardial signal to what would be expected in normal T2-weighted images
3. **Signal pattern identification**: Identify whether signal is normal, uniformly high, uniformly low, or mixed
4. **Edema detection**: High T2 signal typically indicates edema or inflammation
5. **Overall signal classification**: Classify the T2 signal as normal, high, low, or mixed

Example structure: "The T2 signal shows [normal/high/low/mixed] characteristics. [Signal intensity assessment]. [Comparison to normal]. [Edema indicators if high signal]. This indicates [signal classification]."
```

### 3.2 Abnormal Segments (异常节段) - 多选题 - T2_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI. Task: using ONLY the provided image frames, identify which segments show abnormal T2 signal (multi-select).

Strict rules:
- Select only from A/B/C/D; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Basal segment
B. Mid segment
C. Apical segment
D. Apex
```

**Reason Template:**
```
Your reason should include:
1. **Segment identification**: Identify the cardiac segments (basal, mid, apical, apex) visible in the short-axis view
2. **T2 signal abnormality location**: Determine which segments show abnormal T2 signal (high or low signal)
3. **Long-axis distribution**: Describe the distribution of abnormal signal along the long axis of the left ventricle
4. **Segment-by-segment analysis**: Systematically evaluate each segment for presence of abnormal T2 signal
5. **Overall pattern**: Summarize which segments are involved

Example structure: "Abnormal T2 signal is observed in [specific segments]. [Long-axis distribution description]. [Segment-by-segment findings]. The involved segments are [list segments]."
```

### 3.3 Abnormal Regions (异常区域) - 多选题 - T2_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI. Task: using ONLY the provided image frames, identify which wall regions show abnormal T2 signal (multi-select).

Strict rules:
- Select only from A/B/C/D/E/F; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Anterior wall
B. Anteroseptal wall
C. Inferoseptal wall
D. Inferior wall
E. Inferolateral wall
F. Anterolateral wall
```

**Reason Template:**
```
Your reason should include:
1. **Wall region identification**: Identify the six standard wall regions in the short-axis view
2. **T2 signal abnormality mapping**: Map the location of abnormal T2 signal to specific wall regions
3. **Circumferential distribution**: Describe the circumferential distribution of abnormal signal around the LV
4. **Regional signal analysis**: Systematically evaluate each wall region for presence of abnormal T2 signal
5. **Spatial pattern**: Describe the spatial pattern of involvement (e.g., anterior, inferior, lateral, septal)

Example structure: "Abnormal T2 signal is observed in [specific wall regions]. [Circumferential distribution]. [Regional analysis findings]. The involved regions are [list regions]."
```

### 3.4 Signal Distribution (信号分布) - 多选题 - T2_sax

**Prompt Template:**
```
You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI. Task: using ONLY the provided image frames, identify the distribution pattern of abnormal T2 signal (multi-select).

Strict rules:
- Select only from A/B/C/D/E; one or multiple choices allowed.
- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).
- Output your answer first, then provide a brief reason.
- Do NOT use external knowledge or clinical context; rely on the images only.
- Even if uncertain, you must output the most likely choice(s).

Options:
A. Diffuse
B. Linear
C. Patchy
D. Transmural
E. Speckled
```

**Reason Template:**
```
Your reason should include:
1. **Spatial distribution analysis**: Assess how the abnormal T2 signal is distributed spatially across the myocardium
2. **Pattern characteristics**: Identify whether the pattern is diffuse (widespread), linear (line-like), patchy (irregular patches), transmural (full thickness), or speckled (small scattered areas)
3. **Geometric description**: Describe the geometric shape and extent of abnormal signal areas
4. **Continuity assessment**: Evaluate whether abnormal signal is continuous or discontinuous
5. **Overall pattern classification**: Classify the distribution pattern based on spatial characteristics

Example structure: "The abnormal T2 signal shows [pattern type] distribution. [Spatial characteristics]. [Geometric description]. [Continuity assessment]. This indicates [pattern classification]."
```

---

## 总结

每个序列的 prompt template 都遵循以下原则：
1. **明确的角色定义**：VLM for cardiac [sequence type] MRI
2. **严格的任务说明**：使用 ONLY 提供的图像帧
3. **清晰的输出规则**：字母格式、是否多选
4. **Reason 要求**：当 include_reason=True 时，添加 reason template

每个 reason template 都包含：
1. **结构化的分析步骤**：5个关键分析点
2. **具体的观察指导**：告诉模型应该看什么
3. **示例结构**：提供输出格式参考

这样的设计可以引导模型进行更深入、更有针对性的分析。
