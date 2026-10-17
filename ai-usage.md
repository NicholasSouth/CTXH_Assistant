# AI usage declaration: CTXH Assistant

## Member: Nguyễn Thiện Nhân - 24110114

**Tool:** free ClaudeAI (Anthropic). 

The table lists every part of my work in which AI contributed; the prompts in the next section are quoted from my real chat history and shortened where I pasted long files.
| Part | What the AI produced | What I did myself | Files |
|---|---|---|---|
| Tools and data layer | Reference implementation of the 7 tools (`get_document`, `list_documents`, `find_activity`, `convert_points_to_days`, `list_activities`, `register_activity`, `list_registrations`), 3 dataclasses (`ActivityCode`, `Activity`, `Registration`) and the loader and formatting helpers. | Entered the code, wrote the explanatory comments, ran and tested it. The AI then reviewed my comments and corrected several misunderstandings. | `ctxh_server.py` |
| Evaluation | `score_ragas`, `aggregate`, the CSV and summary writer, `--reuse-logs`, the offline self-tests and the Makefile target. | Integrated the code into `eval.py` and ran it. | `eval.py`, `Makefile` |
| Report text and this file | Guiding on what to write for sections 1, 3, 5, my part of section 8 and this `ai-usage.md`, from the repository code and the course PDFs. | Checked the statements against the code and the data confirm. | `report.docx`, `ai-usage.md` |

##
**Prompt that i used:**
| # | Prompt (shortened) | Result |
|---|---|---|
| **1** | “im learning AI coding with Python … My final project topic is ‘Trợ lý CTXH: quy đổi & đăng ký’ … my friend … tasked me with [filling the empty functions in the server file and creating the GitHub repo]” | Explained the task and the professor’s requirements. |
| **2** | “Make me a guiding file for what need to be done, what function i must write, what they do and how should i write them” | Build guide and the 7 tool functions. |
| **3** | “am i understand the code correctly? what do i do next?” (with my commented `ctxh_server.py`) | Corrections to my comments; list of next steps. |
| **6** | Six uploaded log files and the output of `check_logs.py` | Test-set rows Q07–Q14; corrected key format. |
| **7** | “friend fix ctxh_agent or sth and tell me to do this” (with `eval.py` and the task file) | `score_ragas`, `aggregate`, `reused_log`. |
| **8** | “can u explain what eval.py is for?” and “explain what u added line by line, func by func and why” | Explanations, for my understanding. |
| **9** | “base on my prof’s pdf and my friends’ premade can u write a guide on how to accomplish what my friend asked from me?” | Write and outline with guide on what to write for sections 1, 3, 5, my part of 8 and the AI-usage appendix, this report text. |
| **10** | “Okay did i write them correctly? Should i make tables so it’s more organized?” (with this `report.docx`) | Fix the misunderstanding and create tables for the explanations. |