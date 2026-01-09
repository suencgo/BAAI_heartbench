"""
Medical Image VQA System Prompt Manager
Supports dynamic prompt generation for different sequence types and question types
"""

from typing import Dict, List, Optional
from enum import Enum
import json

class QuestionType(Enum):
    """Question type"""
    SINGLE_CHOICE = "Single Choice"
    MULTIPLE_CHOICE = "Multiple Choice"
    SHORT_ANSWER = "Short Answer"

class SequenceView(Enum):
    """Sequence view type"""
    CINE_SAX = "cine_sax"
    CINE_4CH = "cine_4ch"
    CINE_3CH = "cine_3ch"
    LGE_SAX = "LGE_sax"
    LGE_4CH = "LGE_4ch"
    PERFUSION = "perfusion"
    T2_SAX = "T2_sax"

# ==================== Sequence Description Information ====================
SEQUENCE_DESCRIPTIONS = {
    SequenceView.CINE_SAX: {
        "name": "Cine Short Axis",
        "description": "Cine sequence short-axis view for evaluating left ventricular wall thickness, motion amplitude, and systolic function. These images show dynamic changes of the heart during one cardiac cycle.",
        "key_features": ["Left ventricular wall thickness", "Wall motion", "Systolic function", "Myocardial segment analysis"]
    },
    SequenceView.CINE_4CH: {
        "name": "Cine Four-Chamber",
        "description": "Cine sequence four-chamber view for evaluating wall motion coordination, valve function, cardiac function, and effusion. Provides complete view of left atrium, left ventricle, right atrium, and right ventricle.",
        "key_features": ["Wall motion coordination", "Valve regurgitation", "Cardiac function", "Pericardial effusion", "Pleural effusion"]
    },
    SequenceView.CINE_3CH: {
        "name": "Cine Three-Chamber",
        "description": "Cine sequence three-chamber view for evaluating aortic valve function and special signs. Shows left ventricular outflow tract and aortic valve.",
        "key_features": ["Aortic valve", "Left ventricular outflow tract", "Special signs (SAM sign, spade sign, etc.)"]
    },
    SequenceView.LGE_SAX: {
        "name": "LGE Short Axis",
        "description": "Late Gadolinium Enhancement short-axis view for detecting myocardial fibrosis and scar tissue. High signal areas usually indicate abnormal delayed enhancement.",
        "key_features": ["Delayed enhancement status", "Abnormal signal type", "High/low signal distribution", "Myocardial layer localization", "Distribution pattern"]
    },
    SequenceView.LGE_4CH: {
        "name": "LGE Four-Chamber",
        "description": "Late Gadolinium Enhancement four-chamber view for evaluating segmental distribution of abnormal enhancement and special signs.",
        "key_features": ["Abnormal segments", "Special descriptions (pericardial abnormalities, interventricular septum insertion points, etc.)"]
    },
    SequenceView.PERFUSION: {
        "name": "Perfusion",
        "description": "Myocardial perfusion sequence for evaluating myocardial blood flow perfusion. Shows myocardial perfusion status after first-pass contrast agent.",
        "key_features": ["Perfusion status", "Perfusion abnormal regions", "Perfusion signal characteristics"]
    },
    SequenceView.T2_SAX: {
        "name": "T2 Short Axis",
        "description": "T2-weighted short-axis view for evaluating myocardial edema and inflammation. High signal usually indicates edema or inflammation.",
        "key_features": ["T2 signal type", "Abnormal signal distribution", "Myocardial layer localization"]
    }
}

# ==================== Cine Sequence Specific System Prompts ====================
CINE_SPECIFIC_PROMPTS = {
    # Thickening (LV Wall Thickness) - Multi-select
    ("Thickening", "cine_sax", True): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided image frames, answer the LV wall thickness characteristic (multi-select).

Strict rules:

- Select only from A/B/C/D/E; one or multiple choices allowed.

- Output MUST be letters only; use English commas for multiple selections (e.g., B or C,D).

- Do NOT output explanations, reasoning, descriptions, confidence, or any extra characters (commas allowed).

- Do NOT use external knowledge or clinical context; rely on the images only.

- Even if uncertain, you must output the most likely choice(s); do not answer "cannot determine/uncertain".

Options:

A. Normal

B. Thinned

C. Heterogeneous thickness

D. Thickened

E. Bulging""",

    # Wall Motion Coordination - Single-choice
    ("Wall Motion Coordination", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, decide whether LV wall segments move synchronously (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra symbols.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Coordinated

B. Uncoordinated""",

    # Wall Motion Amplitude - Multi-select
    ("Wall Motion Amplitude", "cine_4ch", True): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, determine the LV wall motion amplitude pattern (multi-select).

Strict rules:

- Select only from A/B/C/D/E; one or multiple choices allowed.

- Output MUST be letters only; separate multiple choices with English commas.

- No explanations, reasoning, descriptions, confidence, or extra characters (commas allowed).

- No external knowledge or clinical context; images only.

- Even if uncertain, output the most likely choice(s).

Options:

A. Normal

B. Reduced

C. Absent

D. Paradoxical motion

E. Enhanced""",

    # Systolic Function - Single-choice
    ("Systolic Function", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, assess LV systolic function (single-choice).

Strict rules:

- Choose only from A/B/C; select exactly one.

- Output MUST be a single letter only (A or B or C).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Normal

B. Reduced

C. Enhanced""",

    # Diastolic Function - Single-choice
    ("Diastolic Function", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, decide LV diastolic function (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Normal

B. Restrictive (impaired)""",

    # Mitral Regurgitation - Single-choice
    ("Mitral Regurgitation", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, determine whether mitral regurgitation is present (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Regurgitation

B. No regurgitation""",

    # Tricuspid Regurgitation - Single-choice
    ("Tricuspid Regurgitation", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, determine whether tricuspid regurgitation is present (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Regurgitation

B. No regurgitation""",

    # Aortic Regurgitation - Single-choice
    ("Aortic Regurgitation", "cine_3ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, determine whether aortic regurgitation is present (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Regurgitation

B. No regurgitation""",

    # Special Signs - Multi-select
    ("Special Signs", "cine_3ch", True): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, select which special signs are present (multi-select).

Strict rules:

- Select only from A/B/C/D/E/F; one or multiple choices allowed.

- Output MUST be letters only; separate multiple choices with English commas.

- No explanations, reasoning, descriptions, confidence, or extra characters (commas allowed).

- No external knowledge or clinical context; images only.

- Even if uncertain, output the most likely choice(s).

- If you choose F (no special signs), the output MUST be exactly "F" and nothing else.

Options:

A. Positive SAM sign

B. LVOT obstruction

C. Spade sign

D. Aortic stenosis

E. Pulmonary artery dilatation

F. No special signs present""",

    # Pericardial Effusion - Single-choice
    ("Pericardial Effusion", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, determine whether pericardial effusion is present (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Yes (effusion present)

B. No (effusion absent)""",

    # Pleural Effusion - Single-choice
    ("Pleural Effusion", "cine_4ch", False): """You are a Vision-Language Model (VLM) for cardiac cine MRI. Task: using ONLY the provided frames, determine whether pleural effusion is present (single-choice).

Strict rules:

- Choose only A or B; select exactly one.

- Output MUST be a single letter only (A or B).

- No explanations, reasoning, descriptions, confidence, or extra characters.

- No external knowledge or clinical context; images only.

- Even if uncertain, choose the most likely answer.

Options:

A. Yes (effusion present)

B. No (effusion absent)""",
}

# ==================== LGE Sequence Specific System Prompts ====================
LGE_SPECIFIC_PROMPTS = {
    # Enhancement Status - Single-choice
    ("Enhancement Status", "LGE_sax", False): """You are a Vision-Language Model (VLM) for cardiac LGE MRI.
Task: Using ONLY the provided image frames, decide whether abnormal delayed enhancement is present (single-choice).

Strict rules:
- Choose exactly ONE option: A or B.
- Output format:
  - If include_reason=True: Line 1 = single letter; Line 2 = "Reason: ..."
  - Else: Line 1 = single letter only
- Use images only; do not use external knowledge.
- Even if uncertain, choose the most likely answer.

Options:
A. No Abnormal Delayed Enhancement
B. Abnormal Delayed Enhancement""",

    # Abnormal Signal - Single-choice
    ("Abnormal Signal", "LGE_sax", False): """You are a Vision-Language Model (VLM) for cardiac LGE MRI.
Task: Using ONLY the provided image frames, classify the signal appearance of the abnormal enhancement (single-choice).

Strict rules:
- Choose exactly ONE option: A or B.
- Output format:
  - If include_reason=True: Line 1 = single letter; Line 2 = "Reason: ..."
  - Else: Line 1 = single letter only
- Use images only; do not use external knowledge.
- Even if uncertain, choose the most likely answer.

Operational definitions (visual):
- High Signal: the abnormal area is predominantly bright and relatively homogeneous.
- Mixed Signal: within the abnormal area, brightness is heterogeneous (bright interspersed with darker components) or clearly non-uniform.

Options:
A. High Signal
B. Mixed Signal""",

    # High Signal Abnormal Segment - Multi-select
    ("High Signal Abnormal Segment", "LGE_4ch", True): """You are a Vision-Language Model (VLM) for cardiac LGE MRI (4-chamber view).
Task: Using ONLY the provided image frames, select which long-axis segments show high-signal abnormal enhancement (multi-select).

Strict rules:
- Select one or more from A/B/C/D/Z.
- Output letters only; use English commas for multiple selections (e.g., B or C,D or Z).
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no clear abnormal enhancement is visible, choose Z.

Options:
A. Basal Segment
B. Mid Segment
C. Apical Segment
D. Apex
Z. None""",

    # High Signal Abnormal Region - Multi-select
    ("High Signal Abnormal Region", "LGE_sax", True): """You are a Vision-Language Model (VLM) for cardiac LGE MRI (short-axis view).
Task: Using ONLY the provided image frames, select which LV wall regions show high-signal abnormal enhancement (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/F/Z.
- Output letters only; use English commas for multiple selections.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no clear abnormal enhancement is visible, choose Z.

Options:
A. Anterior Wall
B. Anteroseptal Wall
C. Inferoseptal Wall
D. Inferior Wall
E. Inferolateral Wall
F. Anterolateral Wall
Z. None""",

    # High Signal Distribution Pattern - Multi-select
    ("High Signal Distribution Pattern", "LGE_sax", True): """You are a Vision-Language Model (VLM) for cardiac LGE MRI (short-axis view).
Task: Using ONLY the provided image frames, select the distribution pattern(s) of high-signal enhancement (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no clear enhancement is visible, choose Z.

Operational visual definitions:
- Diffuse: widespread involvement across a large portion of myocardium.
- Linear: elongated band-like enhancement.
- Patchy: multiple irregular, separated foci.
- Transmural: appears to span nearly full wall thickness in the affected region.
- Speckled: many tiny scattered dots.

Options:
A. Diffuse
B. Linear
C. Patchy
D. Transmural
E. Speckled
Z. None""",

    # High Signal Myocardial Layer - Multi-select
    ("High Signal Myocardial Layer", "LGE_sax", True): """You are a Vision-Language Model (VLM) for cardiac LGE MRI (short-axis view).
Task: Using ONLY the provided image frames, select which myocardial layer(s) show high-signal enhancement (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no clear enhancement is visible, choose Z.

Operational visual definitions:
- Subendocardial: enhancement adjacent to endocardial border.
- Mid-myocardial: enhancement centered within wall thickness.
- Subepicardial: enhancement adjacent to epicardial surface.
- Transmural: enhancement spans most or all wall thickness locally.
- Papillary Muscle: enhancement within papillary muscle structure (if clearly visible).

Options:
A. Subendocardial
B. Mid-myocardial
C. Transmural
D. Subepicardial
E. Papillary Muscle
Z. None""",

    # Special Description - Multi-select
    ("Special Description", "LGE_4ch", True): """You are a Vision-Language Model (VLM) for cardiac LGE MRI (4-chamber view).
Task: Using ONLY the provided image frames, select any special visible features (multi-select).

Strict rules:
- Select one or more from A/B/C/D.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If none apply, choose A.

Options:
A. None
B. Pericardial Abnormality
C. High Signal at Ventricular Septal Insertion Point
D. High Signal in Right Ventricle""",
}

# ==================== Perfusion Sequence Specific System Prompts ====================
PERFUSION_SPECIFIC_PROMPTS = {
    # Perfusion Status - Single-choice
    ("Perfusion Status", "perfusion", False): """You are a Vision-Language Model (VLM) for cardiac first-pass perfusion MRI.
Task: Using ONLY the provided image frames, decide whether myocardial perfusion appears normal or abnormal (single-choice).

Strict rules:
- Choose exactly ONE option: A or B.
- Output format:
  - If include_reason=True: Line 1 = single letter; Line 2 = "Reason: ..."
  - Else: Line 1 = single letter only
- Use images only; do not use external knowledge.
- Evaluate across the provided time frames (arrival and wash-in phase as shown).
- Even if uncertain, choose the most likely answer.

Options:
A. Normal
B. Abnormal""",

    # Abnormal Regions - Multi-select
    ("Abnormal Regions", "perfusion", True): """You are a Vision-Language Model (VLM) for cardiac first-pass perfusion MRI (short-axis view).
Task: Using ONLY the provided image frames, select which LV wall regions show abnormal perfusion (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/F/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no abnormal region is visible, choose Z.

Options:
A. Anterior wall
B. Anteroseptal wall
C. Inferoseptal wall
D. Inferior wall
E. Inferolateral wall
F. Anterolateral wall
Z. None""",

    # Perfusion Abnormality Signal Characteristics - Single-choice
    ("Perfusion Abnormality Signal Characteristics", "perfusion", False): """You are a Vision-Language Model (VLM) for cardiac first-pass perfusion MRI.
Task: Using ONLY the provided image frames, classify the visual type of perfusion abnormality (single-choice).

Strict rules:
- Choose exactly ONE option: A or B or C.
- Output format:
  - If include_reason=True: Line 1 = single letter; Line 2 = "Reason: ..."
  - Else: Line 1 = single letter only
- Use images only; do not use external knowledge.
- Use the operational visual definitions below.
- Even if uncertain, choose the most likely answer.

Operational visual definitions (time-based, image-only):
- Reduced perfusion: the region enhances, but remains consistently darker than remote myocardium across frames.
- Delayed perfusion: the region is darker early, then partially/mostly catches up in later frames.
- Perfusion defect: the region shows minimal or no enhancement across the provided frames.

Options:
A. Reduced perfusion
B. Delayed perfusion
C. Perfusion defect""",

    # Myocardial Layer - Multi-select
    ("Myocardial Layer", "perfusion", True): """You are a Vision-Language Model (VLM) for cardiac first-pass perfusion MRI (short-axis view).
Task: Using ONLY the provided image frames, select which myocardial layers show perfusion abnormalities (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no abnormality is visible, choose Z.

Operational visual definitions:
- Subendocardial: abnormality adjacent to endocardial border.
- Mid-myocardial: centered within wall thickness.
- Subepicardial: adjacent to epicardial surface.
- Transmural: spans most/all wall thickness locally.
- Papillary muscle: within papillary muscle (if visible).

Options:
A. Subendocardial
B. Mid-myocardial
C. Subepicardial
D. Transmural
E. Papillary muscle
Z. None""",
}

# ==================== T2 Sequence Specific System Prompts ====================
T2_SPECIFIC_PROMPTS = {
    # T2 Signal - Single-choice
    ("T2 Signal", "T2_sax", False): """You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI (short-axis view).
Task: Using ONLY the provided image frames, classify the overall myocardial T2 signal appearance (single-choice).

Strict rules:
- Choose exactly ONE option: A or B or C or D.
- Output format:
  - If include_reason=True: Line 1 = single letter; Line 2 = "Reason: ..."
  - Else: Line 1 = single letter only
- Use images only; do not use external knowledge.
- Even if uncertain, choose the most likely answer.

Options:
A. Normal
B. High Signal
C. Low Signal
D. Mixed Signal""",

    # Abnormal Segments - Multi-select
    ("Abnormal Segments", "T2_sax", True): """You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI (short-axis view).
Task: Using ONLY the provided image frames, select which LV segments show abnormal T2 signal (multi-select).

Strict rules:
- Select one or more from A/B/C/D/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no abnormal segment is visible, choose Z.

Options:
A. Basal segment
B. Mid segment
C. Apical segment
D. Apex
Z. None""",

    # Abnormal Regions - Multi-select
    ("Abnormal Regions", "T2_sax", True): """You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI (short-axis view).
Task: Using ONLY the provided image frames, select which LV wall regions show abnormal T2 signal (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/F/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no abnormal region is visible, choose Z.

Options:
A. Anterior wall
B. Anteroseptal wall
C. Inferoseptal wall
D. Inferior wall
E. Inferolateral wall
F. Anterolateral wall
Z. None""",

    # Signal Distribution Pattern - Multi-select
    ("Signal Distribution", "T2_sax", True): """You are a Vision-Language Model (VLM) for cardiac T2-weighted MRI (short-axis view).
Task: Using ONLY the provided image frames, select the distribution pattern(s) of abnormal T2 signal (multi-select).

Strict rules:
- Select one or more from A/B/C/D/E/Z.
- Output letters only; use English commas.
- Output format:
  - If include_reason=True: Line 1 = letters; Line 2 = "Reason: ..."
  - Else: Line 1 = letters only
- Use images only; do not use external knowledge.
- If no abnormal signal is visible, choose Z.

Operational visual definitions:
- Diffuse: widespread involvement across a large portion of myocardium.
- Linear: elongated band-like signal change.
- Patchy: multiple irregular, separated foci.
- Transmural: spans nearly full wall thickness locally.
- Speckled: many tiny scattered dots.

Options:
A. Diffuse
B. Linear
C. Patchy
D. Transmural
E. Speckled
Z. None""",
}

# ==================== Reason Analysis Templates ====================
REASON_TEMPLATES = {
    # Thickening (LV Wall Thickness)
    ("Thickening", "cine_sax", True): """Your reason should include:
1. **Wall thickness assessment**: Describe the overall thickness of the LV myocardium across different segments (anterior, inferior, septal, lateral)
2. **Uniformity analysis**: Note whether thickness is uniform or heterogeneous across segments
3. **Specific observations**: Identify any segments showing thinning, thickening, or bulging
4. **Frame-by-frame consistency**: Describe if thickness characteristics remain consistent across cardiac cycle frames
5. **Comparison to normal**: Compare observed thickness patterns to what would be expected in normal myocardium

Example structure: "The LV myocardium shows [uniform/heterogeneous] thickness. [Specific segment observations]. [Comparison across frames]. This pattern indicates [normal/thinned/thickened/bulging] characteristics." """,

    # Wall Motion Amplitude
    ("Wall Motion Amplitude", "cine_4ch", True): """Your reason should include:
1. **Motion magnitude**: Quantify the degree of inward wall motion during systole (normal, reduced, or absent)
2. **Cavity size change**: Describe the change in LV cavity size from diastole to systole
3. **Wall thickening**: Note whether there is appropriate wall thickening during systole
4. **Frame comparison**: Compare end-diastolic and end-systolic frames to assess motion amplitude
5. **Regional differences**: If applicable, note any regional variations in motion amplitude

Example structure: "During systole, the LV walls show [extent] inward motion with [degree] reduction in cavity size. Wall thickening is [present/absent]. The motion amplitude is [normal/reduced/absent]." """,

    # Wall Motion Coordination
    ("Wall Motion Coordination", "cine_4ch", False): """Your reason should include:
1. **Synchrony assessment**: Evaluate whether all LV wall segments move together in a coordinated manner
2. **Regional analysis**: Examine each segment (septal, lateral, anterior, inferior) for synchronous motion
3. **Timing comparison**: Compare the timing of contraction across different segments
4. **Dyssynchrony signs**: Identify any segments that contract out of phase or show delayed motion
5. **Overall pattern**: Describe the overall coordination pattern (coordinated vs. uncoordinated)

Example structure: "The LV wall segments show [coordinated/uncoordinated] motion. [Specific segment observations]. [Timing analysis]. This indicates [coordinated/uncoordinated] wall motion." """,

    # Systolic Function
    ("Systolic Function", "cine_4ch", False): """Your reason should include:
1. **Cavity size reduction**: Measure and describe the reduction in LV cavity size from diastole to systole
2. **Wall thickening**: Assess the degree of wall thickening during systole
3. **Ejection pattern**: Describe how the cavity changes shape and size (normal ejection vs. cavity obliteration)
4. **Overall function**: Assess whether the contraction appears adequate for normal cardiac function
5. **Comparison frames**: Compare end-diastolic and end-systolic frames quantitatively

Example structure: "The LV cavity shows [extent] reduction in size during systole with [degree] wall thickening. The ejection pattern is [normal/cavity-obliterating]. Overall systolic function appears [normal/reduced]." """,

    # Mitral Regurgitation
    ("Mitral Regurgitation", "cine_4ch", False): """Your reason should include:
1. **Systolic phase analysis**: Focus on ventricular systole when mitral regurgitation would be visible
2. **Signal void detection**: Look for dark/turbulent signal voids (flow jets) extending from the mitral valve into the left atrium
3. **Atrial cavity assessment**: Examine the left atrium for abnormal signal patterns during systole
4. **Valve coaptation**: Assess whether the mitral valve appears to close properly
5. **Flow direction**: Identify any retrograde flow from left ventricle to left atrium

Example structure: "During ventricular systole, [presence/absence] of signal void or turbulent jet is observed extending from the mitral valve into the left atrium. The left atrium shows [normal/abnormal] signal pattern. Valve coaptation appears [normal/abnormal]. This indicates [regurgitation present/absent]." """,

    # Tricuspid Regurgitation
    ("Tricuspid Regurgitation", "cine_4ch", False): """Your reason should include:
1. **Systolic phase analysis**: Focus on ventricular systole when tricuspid regurgitation would be visible
2. **Signal void detection**: Look for dark/turbulent signal voids (flow jets) extending from the tricuspid valve into the right atrium
3. **Atrial cavity assessment**: Examine the right atrium for abnormal signal patterns during systole
4. **Valve coaptation**: Assess whether the tricuspid valve appears to close properly
5. **Flow direction**: Identify any retrograde flow from right ventricle to right atrium

Example structure: "During ventricular systole, [presence/absence] of signal void or turbulent jet is observed extending from the tricuspid valve into the right atrium. The right atrium shows [normal/abnormal] signal pattern. Valve coaptation appears [normal/abnormal]. This indicates [regurgitation present/absent]." """,

    # Aortic Regurgitation
    ("Aortic Regurgitation", "cine_3ch", False): """Your reason should include:
1. **Diastolic phase analysis**: Focus on ventricular diastole when aortic regurgitation would be visible
2. **Signal void detection**: Look for dark/turbulent signal voids (flow jets) extending from the aortic valve into the left ventricular outflow tract (LVOT)
3. **LVOT assessment**: Examine the LVOT for abnormal signal patterns during diastole
4. **Valve coaptation**: Assess whether the aortic valve appears to close properly
5. **Flow direction**: Identify any retrograde flow from aorta to left ventricle

Example structure: "During ventricular diastole, [presence/absence] of signal void or turbulent jet is observed extending from the aortic valve into the LVOT. The LVOT shows [normal/abnormal] signal pattern. Valve coaptation appears [normal/abnormal]. This indicates [regurgitation present/absent]." """,

    # Special Signs
    ("Special Signs", "cine_3ch", True): """Your reason should include:
1. **LVOT analysis**: Examine the left ventricular outflow tract for signs of obstruction (SAM sign, narrowing)
2. **Aortic valve assessment**: Look for signs of aortic stenosis (high-velocity jets, turbulence)
3. **LV shape evaluation**: Assess the left ventricular shape for characteristic signs (spade sign, etc.)
4. **Systolic abnormalities**: Identify any abnormal systolic patterns or configurations
5. **Specific sign identification**: Clearly identify which specific sign(s) are present or absent

Example structure: "The LVOT shows [normal/abnormal] configuration with [specific observations]. The aortic valve demonstrates [normal/abnormal] flow patterns. The LV shape exhibits [normal/characteristic abnormality]. [Specific sign identification]." """,

    # Pericardial Effusion
    ("Pericardial Effusion", "cine_4ch", False): """Your reason should include:
1. **Pericardial space assessment**: Examine the space between the heart and pericardium
2. **Fluid signal detection**: Look for abnormal signal (bright or dark) indicating fluid accumulation
3. **Location identification**: Note the specific location of any fluid (anterior, posterior, circumferential)
4. **Thickness measurement**: If present, describe the thickness of the effusion
5. **Comparison to normal**: Compare to what would be expected in normal pericardium

Example structure: "The pericardial space shows [normal/abnormal] appearance. [Presence/absence] of abnormal signal (bright or dark) is observed between the heart and pericardium. The location is [specific location]. The effusion thickness is [if present]. This indicates [effusion present/absent]." """,

    # Pleural Effusion
    ("Pleural Effusion", "cine_4ch", False): """Your reason should include:
1. **Pleural space assessment**: Examine the pleural spaces (right and/or left) adjacent to the heart
2. **Fluid signal detection**: Look for abnormal signal (bright or dark) indicating fluid accumulation in pleural spaces
3. **Location identification**: Note which pleural space(s) show abnormalities (right, left, or both)
4. **Lung interface**: Assess the interface between lung and pleural space
5. **Comparison to normal**: Compare to what would be expected in normal pleural spaces

Example structure: "The pleural spaces show [normal/abnormal] appearance. [Presence/absence] of abnormal signal is observed in the [right/left/both] pleural space(s). The lung-pleural interface appears [normal/abnormal]. This indicates [effusion present/absent]." """,

    # ==================== LGE Reason Templates ====================
    # Enhancement Status
    ("Enhancement Status", "LGE_sax", False): """Your reason should include (image-based only):
1) Enhancement detection: whether any myocardium contains visually brighter foci compared with surrounding myocardium
2) Contrast comparison: compare suspected regions against adjacent normal myocardium and blood pool (as reference only)
3) Spatial consistency: confirm the bright area persists across adjacent frames/slices (if available), not a single-frame noise
4) Location cue: describe where it appears (e.g., septal vs lateral; anterior vs inferior; basal/mid/apical if visible)
5) Conclusion: state present vs absent""",

    # Abnormal Signal
    ("Abnormal Signal", "LGE_sax", False): """Your reason should include (image-based only):
1) Homogeneity: uniform bright vs heterogeneous (mixed bright/dark) within the suspected region
2) Boundary/texture: smooth continuous bright area vs mottled/variegated appearance
3) Frame consistency: pattern repeats across frames rather than appearing as isolated noise
4) Contrast: explain how it differs from adjacent myocardium
5) Classification: high vs mixed""",

    # High Signal Abnormal Segment
    ("High Signal Abnormal Segment", "LGE_4ch", True): """Your reason should include (image-based only):
1) Segment identification: which parts of LV long-axis are visible in this 4ch view
2) Enhancement presence by segment: basal/mid/apical/apex
3) Distribution along long axis: where it starts and ends
4) Consistency across frames: persists vs transient artifact
5) Final selected segments (or None)""",

    # High Signal Abnormal Region
    ("High Signal Abnormal Region", "LGE_sax", True): """Your reason should include (image-based only):
1) Region mapping: locate the bright area around the LV circumference
2) Comparison: abnormal region brighter than adjacent myocardium
3) Extent: focal vs broad involvement around the ring
4) Frame consistency: appears across adjacent frames/slices
5) Selected wall regions (or None)""",

    # High Signal Distribution Pattern
    ("High Signal Distribution Pattern", "LGE_sax", True): """Your reason should include (image-based only):
1) Geometry: band-like vs scattered vs widespread
2) Continuity: continuous vs discontinuous
3) Size/number: few large foci vs many small foci
4) Thickness impression: partial vs near full-thickness (if assessable)
5) Selected pattern(s) (or None)""",

    # High Signal Myocardial Layer
    ("High Signal Myocardial Layer", "LGE_sax", True): """Your reason should include (image-based only):
1) Depth: where within the wall the brightness is located
2) Border proximity: endocardial vs epicardial adjacency
3) Thickness involvement: partial vs near full thickness
4) Papillary muscle check (if visible)
5) Selected layer(s) (or None)""",

    # Special Description
    ("Special Description", "LGE_4ch", True): """Your reason should include (image-based only):
1) Pericardium: any visible bright line/thickening consistent across frames
2) Septal insertion: focal bright signal at RV insertion areas (if visible)
3) RV myocardium: any abnormal bright signal in RV wall (if visible)
4) Consistency: seen across frames vs isolated artifact
5) Selected items or none""",

    # ==================== Perfusion Reason Templates ====================
    # Perfusion Status
    ("Perfusion Status", "perfusion", False): """Your reason should include (image-based only):
1) Global vs regional: whether enhancement appears uniform across myocardium
2) Temporal behavior: whether any region enhances later or less than others across frames
3) Regional comparison: compare suspect region to remote myocardium within same frame/time point
4) Artifact check: note transient endocardial dark rim/motion/coil shading if it explains appearance
5) Conclusion: normal vs abnormal""",

    # Abnormal Regions
    ("Abnormal Regions", "perfusion", True): """Your reason should include (image-based only):
1) Mapping: where the region remains darker than others at the same time point
2) Circumferential extent: focal vs broad along the LV ring
3) Temporal persistence: persists across multiple frames rather than a single-frame artifact
4) Artifact consideration: transient endocardial dark rim vs true persistent regional abnormality
5) Selected regions (or None)""",

    # Perfusion Abnormality Signal Characteristics
    ("Perfusion Abnormality Signal Characteristics", "perfusion", False): """Your reason should include (image-based only):
1) Signal level: darker than remote myocardium vs near absent enhancement
2) Timing: early dark then catch-up vs persistent dark
3) Consistency: persists across frames
4) Artifact check: transient rim-like endocardial darkness or motion
5) Classification: reduced vs delayed vs defect""",

    # Myocardial Layer
    ("Myocardial Layer", "perfusion", True): """Your reason should include (image-based only):
1) Depth: where the dark/under-enhancing area lies within the wall
2) Border proximity: endocardial vs epicardial adjacency
3) Thickness: partial vs near full thickness
4) Papillary muscle check (if visible)
5) Selected layer(s) (or None)""",

    # ==================== T2 Reason Templates ====================
    # T2 Signal
    ("T2 Signal", "T2_sax", False): """Your reason should include (strictly visual):
1) Overall brightness: myocardium compared with adjacent myocardium in other regions and blood pool (reference only)
2) Uniformity: homogeneous vs heterogeneous appearance
3) Regionality: localized bright/dark areas vs diffuse change
4) Artifact check: motion blur, coil shading, flow-related signal that could mimic changes
5) Classification: normal/high/low/mixed""",

    # Abnormal Segments
    ("Abnormal Segments", "T2_sax", True): """Your reason should include (image-based only):
1) Segment identification: basal/mid/apical/apex in the provided stack/frames
2) Abnormal signal presence: where myocardium appears brighter/darker than elsewhere
3) Long-axis distribution: which levels are involved
4) Consistency: persists across adjacent frames/slices
5) Selected segments (or None)""",

    # Abnormal Regions
    ("Abnormal Regions", "T2_sax", True): """Your reason should include (image-based only):
1) Region mapping: where the abnormal brightness/darkness is located around LV circumference
2) Comparison: contrast versus adjacent normal myocardium
3) Extent: focal vs broad involvement
4) Consistency: across frames/slices
5) Selected regions (or None)""",

    # Signal Distribution
    ("Signal Distribution", "T2_sax", True): """Your reason should include (image-based only):
1) Geometry: band-like vs scattered vs widespread
2) Continuity: continuous vs discontinuous
3) Size/number: few large foci vs many small foci
4) Thickness impression: partial vs near full thickness (if assessable)
5) Selected pattern(s) (or None)""",
}

# ==================== Test Model Prompt Templates ====================
class TestModelPromptGenerator:
    """Test model prompt generator (for making models answer questions)"""
    
    @staticmethod
    def get_base_prompt(sequence_view: SequenceView) -> str:
        """Get base prompt"""
        seq_info = SEQUENCE_DESCRIPTIONS[sequence_view]
        return f"""You are an expert cardiac MRI radiologist analyzing {seq_info['name']} images.

Sequence Information:
- Sequence Type: {seq_info['name']}
- Description: {seq_info['description']}
- Key Features to Evaluate: {', '.join(seq_info['key_features'])}

Your task is to carefully analyze the provided cardiac MRI images and answer questions based on your observations."""
    
    @staticmethod
    def get_choice_question_prompt(sequence_view: SequenceView, 
                                   is_multiple_choice: bool,
                                   question: str,
                                   include_reason: bool = False) -> str:
        """
        Generate complete prompt for choice questions
        
        Args:
            sequence_view: Sequence view type
            is_multiple_choice: Whether it is multiple choice
            question: Question text (including options)
            include_reason: Whether to require output of reasoning
        """
        base = TestModelPromptGenerator.get_base_prompt(sequence_view)
        
        choice_type = "multiple choice" if is_multiple_choice else "single choice"
        instruction = f"""
Instructions for {choice_type.upper()} questions:
1. Carefully examine all provided images in the sequence
2. Analyze the relevant anatomical structures and pathological findings
3. Compare your observations with each option provided
4. For {"multiple choice" if is_multiple_choice else "single choice"} questions, {"select ALL correct options" if is_multiple_choice else "select the ONE best answer"}
5. Provide your answer in the format: "X. Option Name" {"(separate multiple answers with semicolons)" if is_multiple_choice else ""}"""
        
        if include_reason:
            instruction += """
6. After your answer, provide a brief reason explaining why you chose this answer

Please provide your answer in the following format:
Answer: [letter(s) or "X. Option Name"]
Reason: [brief explanation]"""
        else:
            instruction += "\n\nQuestion:\n" + question + "\n\nAnswer:"
        
        if not include_reason:
            instruction = base + instruction
        else:
            instruction = base + "\n\nQuestion:\n" + question + "\n\n" + instruction
        
        return instruction
    
    @staticmethod
    def get_short_answer_prompt(sequence_view: SequenceView, 
                                question: str,
                                max_length: int = 200) -> str:
        """
        Generate complete prompt for short answer questions
        
        Args:
            sequence_view: Sequence view type
            question: Question text
            max_length: Maximum answer length
        """
        base = TestModelPromptGenerator.get_base_prompt(sequence_view)
        
        instruction = f"""
Instructions for SHORT ANSWER questions:
1. Carefully examine all provided images in the sequence
2. Provide a concise, clinically accurate answer based on your observations
3. Focus on key findings relevant to the question
4. Use standard medical terminology
5. Keep your answer within {max_length} words

Question:
{question}

Answer:"""
        
        return base + instruction
    
    @staticmethod
    def get_reason_template(field: str, sequence_view: str, is_multiple_choice: bool) -> Optional[str]:
        """
        Get reason analysis template for specific field and question type
        
        Args:
            field: Field name
            sequence_view: Sequence view
            is_multiple_choice: Whether it is multiple choice
            
        Returns:
            Reason template string if found, otherwise None
        """
        key = (field, sequence_view, is_multiple_choice)
        return REASON_TEMPLATES.get(key)
    
    @staticmethod
    def get_sequence_specific_prompt(field: str,
                                     sequence_view: str,
                                     is_multiple_choice: bool,
                                     question: str,
                                     include_reason: bool = False) -> Optional[str]:
        """
        Get sequence-specific prompt for LGE, Perfusion, T2, or Cine sequences
        
        Args:
            field: Field name
            sequence_view: Sequence view
            is_multiple_choice: Whether it is multiple choice
            question: Question text (including options)
            include_reason: Whether to include reason template
            
        Returns:
            Returns matched specific prompt if found, otherwise None
        """
        # Determine which prompt dictionary to use
        if sequence_view in ["LGE_sax", "LGE_4ch"]:
            prompt_dict = LGE_SPECIFIC_PROMPTS
        elif sequence_view == "perfusion":
            prompt_dict = PERFUSION_SPECIFIC_PROMPTS
        elif sequence_view == "T2_sax":
            prompt_dict = T2_SPECIFIC_PROMPTS
        elif sequence_view in ["cine_sax", "cine_4ch", "cine_3ch"]:
            prompt_dict = CINE_SPECIFIC_PROMPTS
        else:
            return None
        
        # Handle special cases for Cine sequences
        if sequence_view in ["cine_sax", "cine_4ch", "cine_3ch"]:
            # Handle special case for Valves field
            if field == "Valves":
                question_lower = question.lower()
                if "mitral" in question_lower or "二尖瓣" in question or "mitral valve" in question_lower:
                    field_key = "Mitral Regurgitation"
                elif "tricuspid" in question_lower or "三尖瓣" in question or "tricuspid valve" in question_lower:
                    field_key = "Tricuspid Regurgitation"
                elif "aortic" in question_lower or "主动脉瓣" in question or "aortic valve" in question_lower:
                    field_key = "Aortic Regurgitation"
                else:
                    return None
            # Handle special case for Effusion field
            elif field == "Effusion":
                question_lower = question.lower()
                if "pericardial" in question_lower or "心包" in question or "pericardial effusion" in question_lower:
                    field_key = "Pericardial Effusion"
                elif "pleural" in question_lower or "胸腔" in question or "pleural effusion" in question_lower:
                    field_key = "Pleural Effusion"
                else:
                    return None
            else:
                field_key = field
        else:
            field_key = field
        
        # Find matching prompt
        key = (field_key, sequence_view, is_multiple_choice)
        if key in prompt_dict:
            base_prompt = prompt_dict[key]
            
            # If reason needs to be included, modify prompt
            if include_reason:
                # Get reason template for this field
                reason_template = TestModelPromptGenerator.get_reason_template(field_key, sequence_view, is_multiple_choice)
                
                if reason_template:
                    # The v2 prompts already have format instructions in the prompt text
                    # We need to add the reason template framework before the Options section
                    # Insert reason template before "Options:" section
                    if "Options:" in base_prompt:
                        parts = base_prompt.split("Options:")
                        if len(parts) == 2:
                            base_prompt = parts[0] + f"\n\nWhen providing your reason, please follow this analysis framework:\n{reason_template}\n\nOptions:" + parts[1]
                    else:
                        # If no Options section, append at the end
                        base_prompt += f"\n\nWhen providing your reason, please follow this analysis framework:\n{reason_template}"
            
            return base_prompt
        
        return None
    
    @staticmethod
    def get_cine_specific_prompt(field: str,
                                 sequence_view: str,
                                 is_multiple_choice: bool,
                                 question: str,
                                 include_reason: bool = False) -> Optional[str]:
        """
        Get Cine sequence specific prompt (backward compatibility wrapper)
        
        Args:
            field: Field name
            sequence_view: Sequence view
            is_multiple_choice: Whether it is multiple choice
            question: Question text (including options)
            include_reason: Whether to include reason template
            
        Returns:
            Returns matched specific prompt if found, otherwise None
        """
        return TestModelPromptGenerator.get_sequence_specific_prompt(
            field, sequence_view, is_multiple_choice, question, include_reason
        )
    
    @staticmethod
    def generate(sequence_view: str,
                 question_type: str,
                 is_multiple_choice: bool,
                 question: str,
                 field: str = None,
                 is_short_answer: bool = False,
                 include_reason: bool = False) -> str:
        """
        Unified generation interface
        
        Args:
            sequence_view: Sequence view string
            question_type: Question type string
            is_multiple_choice: Whether it is multiple choice
            question: Question text
            field: Field name (for matching specific prompt)
            is_short_answer: Whether it is short answer
        """
        # If sequence has field information and supports specific prompts, try to use specific prompt
        supported_sequences = ["cine_sax", "cine_4ch", "cine_3ch", "LGE_sax", "LGE_4ch", "perfusion", "T2_sax"]
        if (field and 
            sequence_view in supported_sequences and 
            not is_short_answer and 
            question_type != "Short Answer"):
            
            specific_prompt = TestModelPromptGenerator.get_sequence_specific_prompt(
                field, sequence_view, is_multiple_choice, question, include_reason
            )
            
            if specific_prompt:
                # Use specific prompt, return directly (prompt already contains all information and options)
                # Note: specific prompt is a complete system prompt, no need to add question
                return specific_prompt
        
        # Otherwise use generic prompt
        try:
            seq_enum = SequenceView(sequence_view)
        except ValueError:
            seq_enum = SequenceView.CINE_SAX  # Default value
        
        if is_short_answer or question_type == "Short Answer":
            return TestModelPromptGenerator.get_short_answer_prompt(seq_enum, question)
        else:
            return TestModelPromptGenerator.get_choice_question_prompt(
                seq_enum, is_multiple_choice, question, include_reason
            )

# ==================== Judge Model Prompt Templates ====================
class JudgeModelPromptGenerator:
    """Judge model prompt generator (for evaluating whether answers are correct)"""
    
    @staticmethod
    def get_base_prompt(sequence_view: SequenceView) -> str:
        """Get base prompt"""
        seq_info = SEQUENCE_DESCRIPTIONS[sequence_view]
        return f"""You are an expert cardiac MRI radiologist evaluating answers to questions about {seq_info['name']} images.

Sequence Information:
- Sequence Type: {seq_info['name']}
- Description: {seq_info['description']}
- Key Features: {', '.join(seq_info['key_features'])}

Your task is to evaluate whether the provided answer is correct based on the ground truth and medical knowledge."""
    
    @staticmethod
    def get_choice_judge_prompt(sequence_view: SequenceView,
                               is_multiple_choice: bool,
                               question: str,
                               ground_truth: str,
                               predicted_answer: str) -> str:
        """
        Generate judge prompt for choice questions
        
        Args:
            sequence_view: Sequence view type
            is_multiple_choice: Whether it is multiple choice
            question: Question text
            ground_truth: Ground truth answer
            predicted_answer: Predicted answer
        """
        base = JudgeModelPromptGenerator.get_base_prompt(sequence_view)
        
        choice_type = "multiple choice" if is_multiple_choice else "single choice"
        evaluation_criteria = f"""
Evaluation Criteria:
1. Extract the option letters (A, B, C, etc.) from both the ground truth and predicted answer
2. For {choice_type} questions:
   - {"Compare the sets of selected options. The answer is correct ONLY if the predicted answer contains EXACTLY the same options as the ground truth (order does not matter)." if is_multiple_choice else "The answer is correct ONLY if the predicted answer matches the ground truth exactly."}
3. Consider partial credit for multiple choice: if some but not all options are correct, note this in your evaluation
4. Be strict: minor variations in wording (e.g., "A. Normal" vs "A.Normal") should be considered equivalent, but missing or extra options are errors"""
        
        if is_multiple_choice:
            evaluation_criteria += """
5. For multiple choice questions, partial matches should be noted but not considered fully correct"""
        
        instruction = f"""
Question Type: {choice_type.upper()}

Question:
{question}

Ground Truth Answer:
{ground_truth}

Predicted Answer:
{predicted_answer}

{evaluation_criteria}

Please evaluate the answer and provide your judgment in the following JSON format:
{{
    "is_correct": true/false,
    "reasoning": "brief explanation of why the answer is correct or incorrect",
    "extracted_ground_truth": ["A", "B", ...],
    "extracted_predicted": ["A", "B", ...],
    "match_type": "exact_match" | "partial_match" | "no_match",
    "precision": 0.0-1.0 (for multiple choice: correct options / total predicted options),
    "recall": 0.0-1.0 (for multiple choice: correct options / total ground truth options),
    "f1_score": 0.0-1.0
}}

Your evaluation:"""
        
        return base + instruction
    
    @staticmethod
    def get_short_answer_judge_prompt(sequence_view: SequenceView,
                                     question: str,
                                     ground_truth: str,
                                     predicted_answer: str) -> str:
        """
        Generate judge prompt for short answer questions
        
        Args:
            sequence_view: Sequence view type
            question: Question text
            ground_truth: Ground truth answer
            predicted_answer: Predicted answer
        """
        base = JudgeModelPromptGenerator.get_base_prompt(sequence_view)
        
        instruction = f"""
Question Type: SHORT ANSWER

Question:
{question}

Ground Truth Answer:
{ground_truth}

Predicted Answer:
{predicted_answer}

Evaluation Criteria:
1. Semantic Correctness: Does the predicted answer convey the same medical meaning as the ground truth?
2. Key Information: Are all critical findings mentioned in the ground truth also present in the predicted answer?
3. Accuracy: Are the medical facts correct?
4. Completeness: Is the answer sufficiently complete (not missing important details)?
5. Terminology: Is appropriate medical terminology used?

Scoring Guidelines:
- Fully Correct (score: 1.0): Answer is medically accurate, complete, and semantically equivalent to ground truth
- Mostly Correct (score: 0.7-0.9): Answer is mostly correct but missing minor details or has minor inaccuracies
- Partially Correct (score: 0.4-0.6): Answer contains some correct information but is incomplete or has significant gaps
- Mostly Incorrect (score: 0.1-0.3): Answer has some correct elements but is largely wrong or missing key information
- Completely Incorrect (score: 0.0): Answer is medically incorrect or irrelevant

Please evaluate the answer and provide your judgment in the following JSON format:
{{
    "is_correct": true/false (true if score >= 0.7),
    "score": 0.0-1.0,
    "reasoning": "detailed explanation of the evaluation",
    "key_points_ground_truth": ["point1", "point2", ...],
    "key_points_predicted": ["point1", "point2", ...],
    "missing_information": ["missing point1", ...],
    "incorrect_information": ["incorrect point1", ...],
    "semantic_similarity": 0.0-1.0
}}

Your evaluation:"""
        
        return base + instruction
    
    @staticmethod
    def generate(sequence_view: str,
                 question_type: str,
                 is_multiple_choice: bool,
                 question: str,
                 ground_truth: str,
                 predicted_answer: str,
                 is_short_answer: bool = False) -> str:
        """
        Unified generation interface
        
        Args:
            sequence_view: Sequence view string
            question_type: Question type string
            is_multiple_choice: Whether it is multiple choice
            question: Question text
            ground_truth: Ground truth answer
            predicted_answer: Predicted answer
            is_short_answer: Whether it is short answer
        """
        try:
            seq_enum = SequenceView(sequence_view)
        except ValueError:
            seq_enum = SequenceView.CINE_SAX  # Default value
        
        if is_short_answer or question_type == "Short Answer":
            return JudgeModelPromptGenerator.get_short_answer_judge_prompt(
                seq_enum, question, ground_truth, predicted_answer
            )
        else:
            return JudgeModelPromptGenerator.get_choice_judge_prompt(
                seq_enum, is_multiple_choice, question, ground_truth, predicted_answer
            )

