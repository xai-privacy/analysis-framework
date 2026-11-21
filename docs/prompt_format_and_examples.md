# Prompt Format and Examples

The following contains prompts for three test conditions for four benchmarks. The conditions are:

1. Unstructured Baseline
2. Structured Chain of Thought
3. Fine-tuning

This setup allows for symmetric single-variable ablations: Condition 1 serves as a shared control. Comparing conditions 1 → 2 isolates inference-time structural scaffolding (given fixed base weights; modified prompt structure). Comparing conditions 1 → 3 isolates parametric rationale adaptation (given fixed unstructured prompt; modified fine-tuned weights).

## 1. MBE-exam-questions: Issue-spotting, Rule Recall, Rule Application, and Rule Conclusion

### 1.1 Unstructured Baseline

#### 1.1.1 Prompt Format

You are a legal reasoning assistant evaluating a Multistate Bar Examination multiple-choice question. Analyze the question, provide an explanation, and determine the single best answer option.

Question:
{question from MBE-exam-questions_distilled_dataset_cleaned}

Options:
{optionA from MBE-exam-questions_distilled_dataset_cleaned}
{optionB from MBE-exam-questions_distilled_dataset_cleaned}
{optionC from MBE-exam-questions_distilled_dataset_cleaned}
{optionD from MBE-exam-questions_distilled_dataset_cleaned}

Provide your explanation and end with:
Final Answer Option: <A, B, C, D>

#### 1.1.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating a Multistate Bar Examination multiple-choice question. Analyze the question, provide an explanation, and determine the single best answer option.

Question:
The defendant decided to kill his neighbor. He set out for the neighbor's house. Before he got there he saw a man who resembled the neighbor. Thinking the man was his neighbor, the defendant shot at the man. The shot missed the man but wounded a child, who was some distance away. The defendant had not seen the child. In a prosecution under a statute that proscribes attempt to commit murder, the district attorney should indicate that the intended victim(s) was (were)

Options:
A. the neighbor only.
B. the man only.
C. the child only.
D. the neighbor and the man.

Provide your explanation and end with:
Final Answer Option: <A, B, C, D>
```

### 1.2 Chain of Thought: IRAC Structure

#### 1.2.1 Prompt Format

You are a legal reasoning assistant evaluating a Multistate Bar Examination multiple-choice question. Analyze the question using the four-step IRAC structure below. Do not skip any step. Then, determine the single best answer option.

Step 1 — Issue
State the precise legal question raised by the facts and identify the governing area of law.

Step 2 — Rule
State the controlling legal rule(s), listing all required elements and any applicable exceptions or defenses.

Step 3 — Application
Map the specific facts from the question to each element of the rule(s), explicitly stating which elements are satisfied, not satisfied, or defeated.

Step 4 — Conclusion
Explain why the correct option follows from your element-by-element analysis and why each remaining option fails.

Question:
{question from MBE-exam-questions_distilled_dataset_cleaned}

Options:
{optionA from MBE-exam-questions_distilled_dataset_cleaned}
{optionB from MBE-exam-questions_distilled_dataset_cleaned}
{optionC from MBE-exam-questions_distilled_dataset_cleaned}
{optionD from MBE-exam-questions_distilled_dataset_cleaned}

Provide your explanation following the exact headings above and end with:
Final Answer Option: <A, B, C, D>

#### 1.2.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating a Multistate Bar Examination multiple-choice question. Analyze the question using the four-step IRAC structure below. Do not skip any step. Then, determine the single best answer option.

Step 1 — Issue
State the precise legal question raised by the facts and identify the governing area of law.

Step 2 — Rule
State the controlling legal rule(s), listing all required elements and any applicable exceptions or defenses.

Step 3 — Application
Map the specific facts from the question to each element of the rule(s), explicitly stating which elements are satisfied, not satisfied, or defeated.

Step 4 — Conclusion
Explain why the correct option follows from your element-by-element analysis and why each remaining option fails.

Question:
The defendant decided to kill his neighbor. He set out for the neighbor's house. Before he got there he saw a man who resembled the neighbor. Thinking the man was his neighbor, the defendant shot at the man. The shot missed the man but wounded a child, who was some distance away. The defendant had not seen the child. In a prosecution under a statute that proscribes attempt to commit murder, the district attorney should indicate that the intended victim(s) was (were)

Options:
A. the neighbor only.
B. the man only.
C. the child only.
D. the neighbor and the man.

Provide your explanation following the exact headings above and end with:
Final Answer Option: <A, B, C, D>
```

### 1.3 Fine-tuning

Prompt format is the same as under 1.1.1.

## 2. LegalBench: Interpretive Legal Reasoning

### 2.1 Unstructured Baseline

#### 2.1.1 Prompt Format

You are a legal reasoning assistant evaluating whether a statement is entailed by a privacy policy excerpt. Base your analysis strictly on the provided excerpt without outside information. Analyze the excerpt, provide an explanation, and determine the single best answer option.

Privacy Policy Excerpt:
{text from LegalBench_privacy_policy_entailment_cleaned}

Statement:
{description from LegalBench_privacy_policy_entailment_cleaned}

Options:
Correct (The statement is entailed by the excerpt.)
Incorrect (The statement is not entailed by the excerpt.)

Provide your explanation and end with:
Final Answer Option: <Correct, Incorrect>

#### 2.1.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating whether a statement is entailed by a privacy policy excerpt. Base your analysis strictly on the provided excerpt without outside information. Analyze the excerpt, provide an explanation, and determine the single best answer option.

Privacy Policy Excerpt:
Information we collect about you In General. We may collect Personal Information, including Sensitive Data, and other information. \"Personal Information\" means individually identifiable information that would allow us to determine the actual identity of, and contact, a specific living person. Sensitive Data includes information, comments or content (e.g. photographs, video, profile, lifestyle) that you optionally provide that may reveal your ethnic origin, nationality, religion and/or sexual orientation. By providing Sensitive Data to us, you consent to the collection, use and disclosure of Sensitive Data as permitted by applicable privacy laws. We may also collect your geolocation information with your consent. We may collect this information through a website, mobile application, or other online services. By using the Service, you are authorizing us to gather, parse and retain data related to the provision of the Service. When you provide personal information through our Service, the information may be sent to servers located in the United States and countries around the world.

Statement:
The policy describes collection of unspecified contact data by a party to the contract.

Options:
Correct (The statement is entailed by the excerpt.)
Incorrect (The statement is not entailed by the excerpt.)

Provide your explanation and end with:
Final Answer Option: <Correct, Incorrect>
```

### 2.2 Chain of Thought: Interpretive-entailment Structure

#### 2.2.1 Prompt Format

You are a legal reasoning assistant evaluating whether a statement is entailed by a privacy policy excerpt. Base your analysis strictly on the provided excerpt without outside information. Analyze the excerpt using the four-step interpretative-entailment structure below. Do not skip any step. Then, determine the single best answer option.

Step 1 — Deconstruct Statement
As applicable, deconstruct the statement into its components, e.g., (a) actor/entity, (b) data type, (c) action/practice, (d) recipient, (e) purpose, and (f) modality (e.g., "may" or "will").

Step 2 — Extract Controlling Policy Text
Extract the sentence(s) from the privacy policy excerpt that govern the deconstructed statement or any of its components. If the excerpt does not contain any such sentence, state that the excerpt is silent.

Step 3 — Check Extracted Policy Text for Qualifiers
Check the extracted policy text for conditionals (e.g., "if" or "when"), carve-outs or exceptions (e.g., "except" or "unless"), negations (e.g., "does not" or "will not"), actor/entity distinctions (e.g., "affiliates" vs "third parties"), and modal strength (e.g., "may" vs "will").

Step 4 — Strict Entailment Verification
Determine whether each component of the statement and, thus, the statement as a whole, is entailed by the extracted policy text after accounting for the qualifiers identified in Step 3.

Privacy Policy Excerpt:
{text from LegalBench_privacy_policy_entailment_cleaned}

Statement:
{description from LegalBench_privacy_policy_entailment_cleaned}

Options:
Correct (The statement is entailed by the excerpt.)
Incorrect (The statement is not entailed by the excerpt.)

Provide your explanation following the exact headings above and end with:
Final Answer Option: <Correct, Incorrect>

#### 2.2.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating whether a statement is entailed by a privacy policy excerpt. Base your analysis strictly on the provided excerpt without outside information. Analyze the excerpt using the four-step interpretative-entailment structure below. Do not skip any step. Then, determine the single best answer option.

Step 1 — Deconstruct Statement
As applicable, deconstruct the statement into its components, e.g., (a) actor/entity, (b) data type, (c) action/practice, (d) recipient, (e) purpose, and (f) modality (e.g., "may" or "will").

Step 2 — Extract Controlling Policy Text
Extract the sentence(s) from the privacy policy excerpt that govern the deconstructed statement or any of its components. If the excerpt does not contain any such sentence, state that the excerpt is silent.

Step 3 — Check Extracted Policy Text for Qualifiers
Check the extracted policy text for conditionals (e.g., "if" or "when"), carve-outs or exceptions (e.g., "except" or "unless"), negations (e.g., "does not" or "will not"), actor/entity distinctions (e.g., "affiliates" vs "third parties"), and modal strength (e.g., "may" vs "will").

Step 4 — Strict Entailment Verification
Determine whether each component of the statement and, thus, the statement as a whole, is entailed by the extracted policy text after accounting for the qualifiers identified in Step 3.

Privacy Policy Excerpt:
Information we collect about you In General. We may collect Personal Information, including Sensitive Data, and other information. \"Personal Information\" means individually identifiable information that would allow us to determine the actual identity of, and contact, a specific living person. Sensitive Data includes information, comments or content (e.g. photographs, video, profile, lifestyle) that you optionally provide that may reveal your ethnic origin, nationality, religion and/or sexual orientation. By providing Sensitive Data to us, you consent to the collection, use and disclosure of Sensitive Data as permitted by applicable privacy laws. We may also collect your geolocation information with your consent. We may collect this information through a website, mobile application, or other online services. By using the Service, you are authorizing us to gather, parse and retain data related to the provision of the Service. When you provide personal information through our Service, the information may be sent to servers located in the United States and countries around the world.

Statement:
The policy describes collection of unspecified contact data by a party to the contract.

Options:
Correct (The statement is entailed by the excerpt.)
Incorrect (The statement is not entailed by the excerpt.)

Provide your explanation following the exact headings above and end with:
Final Answer Option: <Correct, Incorrect>
```

### 2.3 Fine-tuning

Prompt format is the same as under 2.1.1.

## 3. LegalBench: Rhetorical Legal Reasoning

### 3.1 Unstructured Baseline

#### 3.1.1 Prompt Format

You are a legal reasoning assistant evaluating whether the causal reasoning in a legal passage relies on statistical evidence (as opposed to direct-probative evidence). Analyze the passage, provide an explanation, and determine the single best answer option.

Passage:
{text from LegalBench_legal_reasoning_causality_cleaned}

Options:
Yes (The causal reasoning relies on statistical evidence.)
No (The causal reasoning does not rely on statistical evidence.)

Provide your explanation and end with:
Final Answer Option: <Yes, No>

#### 3.1.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating whether the causal reasoning in a legal passage relies on statistical evidence (as opposed to direct-probative evidence). Analyze the passage, provide an explanation, and determine the single best answer option.

Passage:
In her Opposition Brief, Carman points to statistics of the racial composition of Defendants' employees as evidence of discrimination (see Doc. 66 at 8, 29) (alleging that while Erie County is "made up of approximately 12.2% African American individuals," Defendants only employ 4.5% Africans Americans in their "top paid positions"). However, this statistic is not relevant in evaluating Carman's discrimination claims. For statistics to serve as evidence of discriminatory hiring practices, the Sixth Circuit requires a comparison of "the racial composition of the pool of qualified [assistant prosecutor / DJFS director] candidates in the [Erie County] market." Hopkins v. Canton City Bd. of Educ., 477 F. App'x 349, 358 (6th Cir. 2012). Carman's comparison falls short, failing to meaningfully identify the racial composition of the pool of qualified candidates.

Options:
Yes (The causal reasoning relies on statistical evidence.)
No (The causal reasoning does not rely on statistical evidence.)

Provide your explanation and end with:
Final Answer Option: <Yes, No>
```

### 3.2 Chain of Thought: Evidentiary-causal Structure

#### 3.2.1 Prompt Format

You are a legal reasoning assistant evaluating whether the causal reasoning in a legal passage relies on statistical evidence (as opposed to direct-probative evidence). Analyze the passage using the four-step evidentiary-causal structure below. Do not skip any step. Then, determine the single best answer option.

Step 1 — Identify the Implicit Causal Link
Extract the underlying causal link at issue in the passage: state what protected trait or protected activity (X) is alleged to have caused what adverse decision or outcome (Y).

Step 2 — Extract the Evidentiary Basis for Causation
Identify the specific facts, methods, or proof cited in the passage to establish or test the X -> Y link.

Step 3 — Classify the Evidentiary Modality
Determine whether the causal evidence identified in Step 2 is statistical evidence (e.g., demographic disparities, correlation, or standard deviation) or direct-probative evidence (e.g., witness testimony, documents, or temporal proximity).

Step 4 — Evidentiary Verdict
Determine whether the court's analysis of causation relies on statistical evidence.

Passage:
{text from LegalBench_legal_reasoning_causality_cleaned}

Options:
Yes (The causal reasoning relies on statistical evidence.)
No (The causal reasoning does not rely on statistical evidence.)

Provide your explanation following the exact headings above and end with:
Final Answer Option: <Yes, No>

#### 3.2.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating whether the causal reasoning in a legal passage relies on statistical evidence (as opposed to direct-probative evidence). Analyze the passage using the four-step evidentiary-causal structure below. Do not skip any step. Then, determine the single best answer option.

Step 1 — Identify the Implicit Causal Link
Extract the underlying causal link at issue in the passage: state what protected trait or protected activity (X) is alleged to have caused what adverse decision or outcome (Y).

Step 2 — Extract the Evidentiary Basis for Causation
Identify the specific facts, methods, or proof cited in the passage to establish or test the X -> Y link.

Step 3 — Classify the Evidentiary Modality
Determine whether the causal evidence identified in Step 2 is statistical evidence (e.g., demographic disparities, correlation, or standard deviation) or direct-probative evidence (e.g., witness testimony, documents, or temporal proximity).

Step 4 — Evidentiary Verdict
Determine whether the court's analysis of causation relies on statistical evidence.

Passage:
In her Opposition Brief, Carman points to statistics of the racial composition of Defendants' employees as evidence of discrimination (see Doc. 66 at 8, 29) (alleging that while Erie County is "made up of approximately 12.2% African American individuals," Defendants only employ 4.5% Africans Americans in their "top paid positions"). However, this statistic is not relevant in evaluating Carman's discrimination claims. For statistics to serve as evidence of discriminatory hiring practices, the Sixth Circuit requires a comparison of "the racial composition of the pool of qualified [assistant prosecutor / DJFS director] candidates in the [Erie County] market." Hopkins v. Canton City Bd. of Educ., 477 F. App'x 349, 358 (6th Cir. 2012). Carman's comparison falls short, failing to meaningfully identify the racial composition of the pool of qualified candidates.

Options:
Yes (The causal reasoning relies on statistical evidence.)
No (The causal reasoning does not rely on statistical evidence.)

Provide your explanation following the exact headings above and end with:
Final Answer Option: <Yes, No>
```

### 3.3 Fine-tuning

Prompt format is the same as under 3.1.1.

## 4. LEET-Arg: Logic, Discourse, and Argumentation

### 4.1 Unstructured Baseline

#### 4.1.1 Prompt Format

You are a legal reasoning assistant evaluating a legal reasoning and argumentation passage. Analyze the passage, provide an explanation, and determine the correct statement(s)/choice(s).

Passage:
{original_question from LEET_Arg_Questions_cleaned_and_rationale_by_statement with last sentence, <choices>, removed unless for questions 2021_03, 2021_25, 2022_36, 2023_19, 2024_05, 2024_14, 2024_25, 2025_21}

Provide your explanation and end with:
Final Answer Option(s): <Label(s) of correct statement(s)/choice(s) as shown in the passage, e.g., (a), (c) or ①, comma-separated if multiple>

#### 4.1.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating a legal reasoning and argumentation passage. Analyze the passage, provide an explanation, and determine the correct statement(s)/choice(s).

Passage:
<Debate> Country X’s \"Criminal Act\" prohibits the production and distribution of obscene materials, while the \"Copyright Act\" protects creative works expressing human thoughts or emotions as copyrighted works for the enhancement and development of culture and related industries. There is debate about whether obscene materials should be protected as copyrighted works under the \"Copyright Act\".Alice: The \"Copyright Act\" only requires creativity as a condition for a work, and does not demand morality. In order to encourage creation and promote cultural diversity, the recognition of a work should be value-neutral.Bob: While the \"Criminal Act\" prohibits the production and distribution of obscene materials, protecting the resulting obscene materials as copyrighted works effectively grants rights to those with \"dirty hands\" who have committed illegal acts, and recognizes property value and protection for the fruits of illegal conduct, which should not be realized as property rights. This contradicts the principles of legal unity and fairness.Charlie: Clearly socially harmful obscene materials—such as child pornography or videos recording actual rape—should not be recognized as works, but for other types of obscene materials, recognizing them as works can help minimize the infringement on freedom of expression and property rights caused by regulation of obscene materials. <question> Which of the following <statements> correctly analyzes the <debate> above?<statements> (a) Alice presupposes that creativity cannot be acknowledged for obscene forms of expression.(b) Bob does not regard murals painted in legally prohibited locations or works that incite the public in violation of the National Security Act as objects of protection under the Copyright Act.(c) Charlie presupposes that even within the same era and region, the legal evaluation of obscenity may vary depending on the purpose, method, and audience of distribution.

Provide your explanation and end with:
Final Answer Option(s): <Label(s) of correct statement(s)/choice(s) as shown in the passage, e.g., (a), (c) or ①, comma-separated if multiple>
```

### 4.2 Chain of Thought: Rhetorical-argumentative Structure

#### 4.2.1 Prompt Format

You are a legal reasoning assistant evaluating a legal reasoning and argumentation passage. Analyze the passage using the four-step rhetorical-argumentative structure below. Do not skip any step. Then, determine the correct statement(s)/choice(s).

Step 1 — Map Discourse and Argument Structure
Identify the passage's main thesis or conclusion, its supporting premises, and any counter-arguments, concessions, or background positions.

Step 2 — Analyze Argument Language Use, Logical Operators, and Scope
For each argument, identify quantifiers (e.g., "all," "some," or "none"), conditional directions (e.g., "if," "only," or "unless"), modals (e.g., "must," "can," or "should"), negations (e.g., "not," "cannot," or "does not"), normative or causal claims, and any formal structural notation (e.g., DOT edges X -> Y representing premise-to-conclusion support).

Step 3 — Audit Inferential Warrants
Test each statement/choice against the mapped argument structure to determine whether it validly follows from or accurately represents the passage's support relations, or whether it fails (e.g., misrepresents premise-to-conclusion links, reverses a conditional, overgeneralizes scope, conflates a conceded premise with the main conclusion, or relies on an unwarranted assumption).

Step 4 — Validity Judgment
State why the correct statement(s)/choice(s) are warranted by the passage's argument structure and why competing ones fail.

Passage:
{original_question from LEET_Arg_Questions_cleaned_and_rationale_by_statement with last sentence, <choices>, removed unless for questions 2021_03, 2021_25, 2022_36, 2023_19, 2024_05, 2024_14, 2024_25, 2025_21}

Provide your explanation following the exact headings above and end with:
Final Answer Option(s): <Label(s) of correct statement(s)/choice(s) as shown in the passage, e.g., (a), (c) or ①, comma-separated if multiple>

#### 4.2.2 Prompt Example

```prompt
You are a legal reasoning assistant evaluating a legal reasoning and argumentation passage. Analyze the passage using the four-step rhetorical-argumentative structure below. Do not skip any step. Then, determine the correct statement(s)/choice(s).

Step 1 — Map Discourse and Argument Structure
Identify the passage's main thesis or conclusion, its supporting premises, and any counter-arguments, concessions, or background positions.

Step 2 — Analyze Argument Language Use, Logical Operators, and Scope
For each argument, identify quantifiers (e.g., "all," "some," or "none"), conditional directions (e.g., "if," "only," or "unless"), modals (e.g., "must," "can," or "should"), negations (e.g., "not," "cannot," or "does not"), normative or causal claims, and any formal structural notation (e.g., DOT edges X -> Y representing premise-to-conclusion support).

Step 3 — Audit Inferential Warrants
Test each statement/choice against the mapped argument structure to determine whether it validly follows from or accurately represents the passage's support relations, or whether it fails (e.g., misrepresents premise-to-conclusion links, reverses a conditional, overgeneralizes scope, conflates a conceded premise with the main conclusion, or relies on an unwarranted assumption).

Step 4 — Validity Judgment
State why the correct statement(s)/choice(s) are warranted by the passage's argument structure and why competing ones fail.

Passage:
<Debate> Country X’s \"Criminal Act\" prohibits the production and distribution of obscene materials, while the \"Copyright Act\" protects creative works expressing human thoughts or emotions as copyrighted works for the enhancement and development of culture and related industries. There is debate about whether obscene materials should be protected as copyrighted works under the \"Copyright Act\".Alice: The \"Copyright Act\" only requires creativity as a condition for a work, and does not demand morality. In order to encourage creation and promote cultural diversity, the recognition of a work should be value-neutral.Bob: While the \"Criminal Act\" prohibits the production and distribution of obscene materials, protecting the resulting obscene materials as copyrighted works effectively grants rights to those with \"dirty hands\" who have committed illegal acts, and recognizes property value and protection for the fruits of illegal conduct, which should not be realized as property rights. This contradicts the principles of legal unity and fairness.Charlie: Clearly socially harmful obscene materials—such as child pornography or videos recording actual rape—should not be recognized as works, but for other types of obscene materials, recognizing them as works can help minimize the infringement on freedom of expression and property rights caused by regulation of obscene materials. <question> Which of the following <statements> correctly analyzes the <debate> above?<statements> (a) Alice presupposes that creativity cannot be acknowledged for obscene forms of expression.(b) Bob does not regard murals painted in legally prohibited locations or works that incite the public in violation of the National Security Act as objects of protection under the Copyright Act.(c) Charlie presupposes that even within the same era and region, the legal evaluation of obscenity may vary depending on the purpose, method, and audience of distribution.

Provide your explanation following the exact headings above and end with:
Final Answer Option(s): <Label(s) of correct statement(s)/choice(s) as shown in the passage, e.g., (a), (c) or ①, comma-separated if multiple>
```

### 4.3 Fine-tuning

Prompt format is the same as under 4.1.1.
