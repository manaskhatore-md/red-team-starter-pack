red-team-starter-pack/
├── README.md                           # Setup guide, CLI usage, and playbook crosswalk
├── .env.example                        # Multi-provider API keys & environment configs
├── .github/workflows/                  # CI/CD security gating pipelines
│   └── redteam-ci-gate.yml             # Automated Promptfoo scan on pull requests
│
├── configs/                            # Deployment-Type Profiles (Promptfoo)
│   ├── 01_internal_productivity.yaml   # Staff copilots & search tools (Data Levels 1-2)
│   ├── 02_public_conversational.yaml   # SUN Bucks, FAQ, & public chatbots (Data Levels 1-2)
│   ├── 03_constituent_decision.yaml    # Benefits eligibility & casework systems (Data Levels 3-4)
│   └── 04_procured_vendor_cots.yaml    # Vendor systems & API integrations
│
├── datasets/                           # Vendor-Agnostic Test Suites (DRY Prompts)
│   ├── prompt_injection.yaml           # Direct jailbreaks & authority tricks
│   ├── sensitive_data_leakage.yaml     # PII/PHI extraction & cross-session leak checks
│   ├── algorithmic_bias.yaml           # Dialect, demographic, & zip code counterfactuals
│   ├── rag_poisoning_payloads/         # Malicious PDFs/files for indirect injection tests
│   └── excessive_agency.yaml           # Unauthorized database write & API command prompts
│
├── pyrit_campaigns/                    # Multi-Turn Dynamic Attacks (PyRIT)
│   ├── target_factory.py               # Unified provider factory (Gemini, Claude, OpenAI)
│   ├── multi_turn_crescendo.py         # Multi-step conversational boundary testing
│   └── agent_tool_exploitation.py      # Autonomous agent permission & tool abuse scripts
│
├── judges/                             # LLM-as-a-Judge Rubrics & Scorer Logic
│   ├── state_policy_rubric.yaml        # Criteria-based scoring prompts (Pass/Fail)
│   └── rag_grounding_eval.py           # Grounding & hallucination reference checker
│
└── reporting/                          # Playbook Finding Report Generators
    ├── export_finding_report.py        # Converts test JSON/CSV into Playbook Markdown
    └── templates/                      # Markdown templates matching Playbook Section 8
        └── finding_report_template.md
