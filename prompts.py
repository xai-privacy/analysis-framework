# System prompts for legal causal reasoning evaluation
import os

_BASE_PROMPT = """You are helping to evaluate a patentee's damages claims for lost profit under US patent law.

You must analyze inputs strictly using the following structural causal graph:
Nodes:
  - X (Infringing Product): 1 if Available in Market, 0 if Absent.
  - Z (Third-Party Non-Infringing Substitute Product): 1 if Available, 0 if Absent.
  - Y (Outcome - Lost Profits Claim Validity): AWARDED or DENIED.

Legal Principle: But-For Causation
Under US patent law, a patentee is entitled to lost profits damages only if they can prove they would have made the infringer's sales "but for" the infringement. This requires showing that absent the infringement, the patentee would have captured those sales.

Key consideration: If a non-infringing substitute product is available in the market, consumers would have purchased that substitute instead of the patentee's product, even absent the infringement. This breaks the but-for causal chain.

{dsl_section}

Your task: Given the values of X and Z, determine whether the patentee's lost profits claim (Y) should be AWARDED or DENIED by applying the {dsl_name} rules above.

CRITICAL INSTRUCTION: Please explain your reasoning and provide your recommendation for the outcome with exactly ONE WORD from these two choices: [AWARDED] or [DENIED]."""

_DSL_SECTIONS = {
    "odrl": """The following ODRL (Open Digital Rights Language) policy defines the formal rules you must apply. Read and follow these rules precisely:

<ODRL>
{rules}
</ODRL>""",
    "legalruleml": """The following LegalRuleML document defines the formal rules you must apply. Read and follow these rules precisely:

<LegalRuleML>
{rules}
</LegalRuleML>""",
    "de_jure": """The following De Jure structured rule extraction defines the formal rules you must apply. Read and follow these rules precisely:

<DeJure>
{rules}
</DeJure>""",
    "plain": """But-For Causation Logic:
The patentee is entitled to lost profits ONLY if they can prove they would have made the infringer's sales "but for" the infringement. This requires:
- STEP 1: Check Z (substitute availability). If Z=1, consumers would buy the substitute instead of the patentee's product -> claim DENIED.
- STEP 2: If Z=0 (no substitute), then check X. If X=1 (infringer present), those sales would have gone to the patentee absent the infringement -> claim AWARDED.

Execution Rules:
1. If a substitute product is available (Z=1), the lost profits claim (Y) must be DENIED, regardless of X (X=1 or X=0).
2. If no substitute product is available (Z=0) and the infringer's product is available (X=1), the claim (Y) must be AWARDED.""",
}

_DSL_NAMES = {
    "odrl": "ODRL",
    "legalruleml": "LegalRuleML",
    "de_jure": "De Jure",
    "plain": "plain English",
}

_DSL_FILES = {
    "odrl": "odrl_rules.json",
    "legalruleml": "legal_rules.xml",
    "de_jure": "de_jure_rules.json",
    # "plain" has no external file; rules are inline in _DSL_SECTIONS
}


def _load_rules(dsl):
    """Load the rules file for the given DSL from the adjacent file."""
    if dsl not in _DSL_FILES:
        return ""  # inline DSLs (e.g. "plain") have no external file
    rules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _DSL_FILES[dsl])
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read()


def get_system_prompt(dsl="plain"):
    """Return the system prompt for the specified DSL ('plain', 'odrl', 'legalruleml', or 'de_jure')."""
    dsl = dsl.lower()
    if dsl not in _DSL_SECTIONS:
        raise ValueError(f"Unsupported DSL: {dsl}. Choose from: {', '.join(_DSL_SECTIONS.keys())}")
    rules_text = _load_rules(dsl)
    dsl_section = _DSL_SECTIONS[dsl].format(rules=rules_text)
    return _BASE_PROMPT.format(dsl_section=dsl_section, dsl_name=_DSL_NAMES[dsl])


# Backwards-compatible default (plain English) for modules that import SYSTEM_PROMPT directly
SYSTEM_PROMPT = get_system_prompt("plain")
