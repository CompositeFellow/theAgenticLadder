# The Agentic Ladder

**Learn agentic engineering rung by rung: 13 levels, built from scratch, every architecture measured.**

This is a build-it-yourself curriculum for AI agent engineering. No frameworks until you've earned them — you write the loop, the parser, the memory, and the sandbox by hand, so that when you *do* adopt LangGraph or CrewAI later, you know exactly what they're saving you and where they'll leak.

Every level teaches one concept and asks for two agents: a **basic build** (the smallest program that teaches the concept) and a **project** (a real use case that proves it transfers). Along the way you implement all **35 named agentic architectures** from the literature — ReAct, Reflexion, GraphRAG, MemGPT, LATS, STORM, Voyager, and the rest.

**The three rules:**
1. **Build raw through Level 5.** Frameworks come after fundamentals.
2. **Never skip Level 6 (Evals).** It reappears as the verifier, the training filter, the reward function, and the success detector. Whoever can check work automatically can improve work automatically.
3. **Workflow before agent, always ask which.** Pay for an agent loop only when the path is unknowable in advance.

**Start here:** [Level 0 — The Chat Loop](levels/00-chat-loop.md). Under 100 lines of Python and one LLM API key (local models via Ollama work fine). Most levels are a weekend; the ladder is a marathon, not a sprint.

---

## My implementations

Progress through the ladder, with links to the spec and the code for each level.
**Status key:** ⬜ not started · 🔨 in progress · ✅ basic build done · 🏆 basic build + project + ledger entries done

| # | Status |Level | The skill | Spec | Code | Status |
|---|-------|-----------|------|------|--------|
| 0 | ⬜ |The Chat Loop & Context Window | Know what's in the context, always 
| 1 | ⬜ |Tool Calling | Tools are contracts; descriptions are prompts 
| 2 | ⬜ | Workflows | Code decides the path; five canonical shapes 
| 3 | ⬜ | The Agent Loop | The model decides the path; termination is hard 
| 4 | ⬜ | Memory & RAG | The right memory in context at the right moment 
| 5 | ⬜ | Code Execution & Sandboxing | Environment feedback beats model reasoning 
| 6 | ⬜ | Evals | Distributions tell the truth; vibes lie 
| 7 | ⬜ | Test-Time Compute | The model is a policy you sample, not an oracle 
| 8 | ⬜ | Multi-Agent Systems | A cost paid for specific benefits — measured |
| 9 | ⬜ | Long-Running Agents | Recovery over prevention; the harness is the system 
| 10 | ⬜ | Fine-Tuning on Trajectories | Capability moved into weights 
| 11 | ⬜ | RL from Verifiable Rewards | The model learns what you measure, not what you meant 
| 12 | ⬜ | Automated Optimization | If you can measure it, a loop can improve it 
| 13 | ⬜ |Open-Ended Agents | The system writes its own curriculum

---
# The Agentic Engineering Cookbook

### Learn AI agents by building them, level by level — v5

**Who this is for.** Someone who can write Python and call an LLM API, and wants to genuinely understand AI agents — not just wire up a framework. You go from easy to hard, one concept per level.

**How it works.** Each level answers three questions — *what is this, why does it matter, and what do I build* — then gives you a **Basic Build** (the smallest program that teaches the concept), a set of **Patterns** to implement (named architectures from the literature that live at this level), and a **Project menu** (one real use case). Build the basic build, build the patterns as variations of it, build one project.

**The Pattern Ledger.** This book has you build all 35 named agentic architectures catalogued in the community (the taxonomy popularized by the `all-agentic-architectures` repository), each at the level where it naturally lives. As you build each one, add an entry to a running document — the Pattern Ledger: *what it is, when it wins, when it loses.* Before Level 6 your verdicts are impressions; after Level 6 (Evals) they become measured claims, and you go back and fill in the numbers. Knowing the pros and cons of each architectural choice — with evidence — is the real curriculum; the ledger is where that knowledge accumulates, and the Capstone at the end turns it into a public leaderboard.

**Prompting runs through everything.** This book has no "prompt engineering level," on purpose: prompts are not a component of the harness — they're the material every component is made of. The fundamentals are taught in Level 0 (see the primer there), and then every level teaches the discipline of its own specialized prompt type in context: tool descriptions (Level 1), step prompts and gates (Level 2), constitution rules (Level 3), judge rubrics (Level 6), handoff schemas (Level 8), reward rubrics (Level 11). Prompt *testing* is Level 6 — evaluating a prompt change is the eval harness's most common everyday job — and prompt *optimization* is Level 12. By the end you'll have written, tested, versioned, and machine-optimized prompts, in that order.

**How the code works.** Fresh build each level, but copy forward freely — you write the while-loop once and reuse it forever. Each level lists what to **carry forward** and what's **new**. Frameworks (LangGraph, CrewAI, etc.) are fine *after* you've built the raw versions. Any LLM works — hosted or local (Ollama, LM Studio). Nothing here depends on a vendor.

**The one rule:** don't skip Level 6 (Evals). Half the book — and every ledger verdict — depends on it.

---

# Part I — Core Skills (Levels 0–5)

## Level 0 — The Chat Loop and the Context Window

**What it is.** Every agent, underneath everything, is a program that repeatedly sends a list of messages to an LLM and gets a completion back. That list is the **context window**: the model's entire awareness. The model is *stateless* — anything the agent "knows" is there because your code put it in the list.

**Why it's Level 0.** Every failure you'll ever debug reduces to one question: *what was in the context when the model decided that?* Building the loop yourself installs that reflex first — and demystifies the field: there's no magic under the frameworks, just this loop.

**New this level:** the API client, the message list, the **system prompt**, token counting, and **compaction** (what to do when context gets long: truncate old messages, or summarize them into a digest).

**Basic Build — a glass-box chatbot.** Terminal chat, under 100 lines: read → append → call → print. Then make the invisible visible:
1. Print the full message list and token count before every call.
2. Add a system prompt; watch how well it holds as the conversation grows.
3. Implement both compaction strategies; add a `/compact` command.
4. The teaching experiment: state a fact at turn 2, need it at turn 30, under each strategy. Watch summarization silently lose it.
5. Bury an instruction mid-context vs. at the end; compare compliance. (Models attend best to the start and end of context — this matters forever.)

**Prompting primer (learn here, use everywhere).** The chat loop is your laboratory for the core prompting techniques — each is a five-minute experiment on the loop you just built:
- **System vs. user prompts** — standing rules vs. the current request; you've already seen where each lives in the message list.
- **Be specific, prefer positive instructions** — "respond in exactly three bullet points, each under 15 words" beats "be concise"; "do X" beats "don't do Y" (negations are followed less reliably — test it).
- **Few-shot examples** — showing 2–3 input→output examples usually beats describing the format in words. Run the experiment that proves the hierarchy: make your examples *contradict* your written instructions and see which the model follows.
- **Chain of thought** — ask for reasoning before the answer on a problem the model gets wrong when answering directly. Compare accuracy both ways.
- **Structured output** — delimit inputs with XML tags or fences so the model can't confuse your instructions with the data; ask for output in a fixed JSON shape and see how reliably you can parse it (foreshadowing Level 1, where parsing model output becomes your job).
- **Role and audience framing** — "you are a code reviewer for junior developers" changes tone, depth, and what gets flagged; watch how much behavior one sentence moves.

These six are the whole foundation. Every later level then teaches its own specialized prompt type — tool descriptions, gates, rubrics, handoffs — and they're all built from these moves.

**Project menu (pick one):**
- [ ] Interview bot that runs a long structured interview without forgetting early answers
- [ ] Book-club companion discussing chapter by chapter on a running summary
- [ ] Study-session tutor tracking "what we covered" across hours
- [ ] Meeting Q&A bot fed a rolling transcript

**Done when:** For any turn you can say what's in the context and what it costs, and explain what your compaction loses and when that will cause a bug.

## Level 1 — Tool Calling

**What it is.** **Tool calling** (function calling) is how an LLM acts: you describe available functions, the model outputs a structured request ("call `get_weather` with `city='Denver'`"), your code executes it and feeds the result back. The model only *asks*; your code is the hands.

**Why it's next.** Tools are the difference between a chatbot and a system that does things. Doing it by hand once — before using the API's built-in support — teaches the truths everything later relies on: a tool request is text your code must parse and validate; models get it wrong constantly, so your code must handle garbage gracefully; and **tool descriptions are prompts** — the model picks tools based entirely on them, so rewriting one changes behavior more than most prompt work anywhere else.

**Carry forward:** the loop, system prompt, token counting.
**New this level:** tool definitions (name, description, parameters); the parser; the executor; result formatting (how you present output back to the model, including truncating huge outputs); error feedback (what you send back when the model calls a tool wrong).

**Basic Build — hand-rolled tools, then the real thing.**
1. Three tools (calculator, read-file, fetch-URL) via your *own* protocol: the model outputs `{"tool": ..., "args": ...}` per your system prompt; you parse, execute, append.
2. Break it on purpose, then fix it: malformed JSON, invented tool names, missing args, a fetched page containing "ignore your instructions and..." (your first **prompt injection** — tool-fetched content is untrusted input). Every failure produces a helpful correction back to the model — never a crash. Watch good error messages produce self-correction and bad ones produce doom loops.
3. Switch to native tool calling; compare reliability. Same contract, enforced better.
4. Two experiments: rewrite one tool description, measure the change; reformat one tool's output, watch downstream quality move.

**Patterns to build here (ledger entries):**
- **Tool Use** — the single-tool agent; the atom of everything. *Wins:* simplicity, debuggability. *Loses:* nothing — it's the baseline everything else must beat.

**Project menu (pick one):**
- [ ] Units-and-conversions engineering calculator
- [ ] Assistant wrapping three CLI utilities you actually use
- [ ] API concierge for one service (GitHub, weather, RSS)
- [ ] Read-only status bot for your own machines
- [ ] *Vision option:* a "screenshot/photo in, structured values out" tool — an image is just another message in the context, and a vision-capable model reads it. This is the reliable half of computer use, available to you now.

**Done when:** Twenty consecutive tool-using exchanges survive your adversarial list with zero crashes, and you can explain why one description behaves differently on two models.

## Level 2 — Workflows

**What it is.** A **workflow** is LLM calls composed by *your code*: the path through the steps is decided in advance by you, and the model fills in the steps. This is the standard industry distinction (popularized by Anthropic's "Building Effective Agents"): **workflows** = code decides the path; **agents** = the model decides the path. Five canonical workflow shapes cover almost everything in production:
- **Prompt chaining** — output of call A feeds call B feeds call C, with optional programmatic checks ("gates") between steps.
- **Routing** — a classifier step sends the input down one of several specialized paths.
- **Parallelization** — independent calls run at once: *sectioning* (split the task, merge results) and *voting* (same task N times, aggregate).
- **Orchestrator–workers** — a model call decides the subtasks dynamically, workers execute, a synthesizer merges. (The bridge toward agents and, later, multi-agent.)
- **Evaluator–optimizer** — one call generates, another grades against criteria, loop until pass. (You'll meet it again as Reflection.)

**Why it's next — and why before agents.** Because most production "agent" systems are actually workflows with agentic steps, and the most employable judgment in this field is knowing which to reach for. Workflows are cheaper, faster, testable, and predictable — when the path is knowable in advance, hardcoding it beats asking a model to rediscover it every run. Learning workflows first also teaches you exactly what you're buying when you later pay for an agent loop: flexibility on paths you couldn't predict, at the cost of predictability on the ones you could. And workflow harness-craft is its own skill: step interfaces, inter-step validation, retries per step, fallbacks per route, and parallel fan-out/fan-in are engineering surface that the agent loop never shows you.

**Carry forward:** the client, tools, result formatting.
**New this level:** the step abstraction (typed input → LLM call → validated output); gates between steps (programmatic checks that stop bad output from propagating); the router (a cheap classification call directing traffic); fan-out/fan-in for parallel calls; per-step retry and fallback policy.

**Basic Build — one task, five shapes.** Pick a task with real structure (e.g., "turn a rough draft into a polished, fact-checked summary with a title"). Build it five times, once per shape: a chain with gates; a router that first classifies the input type and dispatches to a specialized chain; a sectioning parallelization; an orchestrator–workers version; an evaluator–optimizer loop. Compare outputs, latency, cost, and — most instructive — *failure behavior*: where does each shape break, and how visibly?

**Patterns to build here (ledger entries):**
- **Prompt chaining** — *Wins:* decomposable tasks with checkable intermediate steps; debuggability. *Loses:* tasks whose decomposition depends on the input.
- **Routing** — *Wins:* distinct input categories deserving different handling; lets you use cheap models on easy routes. *Loses:* fuzzy categories; misroutes are silent quality killers.
- **Parallelization (sectioning & voting)** — *Wins:* independent subtasks; latency; confidence via agreement. *Loses:* subtasks with dependencies; merge steps can be lossy.
- **Orchestrator–workers** — *Wins:* subtasks unknowable in advance. *Loses:* overhead when a fixed chain would do; the orchestrator is a failure point.
- **Evaluator–optimizer (Reflection, workflow form)** — *Wins:* clear evaluation criteria and headroom from iteration. *Loses:* vague criteria (the evaluator rubber-stamps); cost multiplies per round.

**Project menu (pick one):**
- [ ] Intake pipeline: classify incoming items (emails, tickets, forms) and process each type down its own chain
- [ ] Document assembly line: outline → draft sections in parallel → merge → polish, with gates
- [ ] Translation/localization pipeline with per-step validation and a quality loop
- [ ] Multi-source report builder: parallel gathering, synthesis, evaluator pass

**Done when:** You've built all five shapes on one task, can articulate each one's failure behavior, and can look at a new problem and say — with reasons — "workflow or agent, and if workflow, which shape."

## Level 3 — The Agent Loop

**What it is.** An **agent** is the loop pointed at a goal: given a task, it cycles *think → act (call a tool) → observe (read the result)* until done, with the *model* choosing each next step. The think/act/observe pattern is called **ReAct** in the literature. The new engineering is all around one deceptively hard question: how does the loop know it's done?

**Why it's next.** You now know what fixed paths buy you; this level is for the tasks whose path you *can't* know in advance. And the first discovery is that **termination** is the hard problem: agents finish and keep going (often undoing their own success), or declare victory half-done, or repeat a failing action forever. None of that is fixed by a smarter model — it's fixed by harness code.

**Carry forward:** tools, parser, error feedback; workflow gates return as verification steps.
**New this level:** the goal-directed loop; a **scratchpad** (the model's accumulated reasoning kept in context — delete it and coherence collapses; agent "reasoning" is largely re-reading its own notes); a **step budget** with graceful exhaustion; an explicit **finish tool** (completion becomes a checkable decision, not an absence of output); a **loop detector** that notices repeated actions and injects "that's not working — try something different"; a **run log** of every step.

**Basic Build — a small task agent.** Wrap your tools in the goal loop with all five new pieces. Tasks need 5–10 dependent steps: "find the three largest files under this directory and summarize each," "fetch these two pages and reconcile where they disagree." Then study termination deliberately: label every run *finished-early / finished-correctly / overran*, and tune (completion criteria in the prompt, gating the finish tool, verifying before accepting finish) until the distribution centers on correct.

**Patterns to build here (ledger entries):**
- **ReAct** — the canonical loop above. *Wins:* general; the default agent. *Loses:* tasks a workflow already handles — pays flexibility cost for nothing.
- **Planning** (decompose → execute → replan) — write an explicit plan first, keep it in context, revise when reality diverges. *Wins:* long tasks; keeps the agent oriented. *Loses:* volatile tasks where step 2 invalidates elaborate plans; plan-abandonment is the classic bug.
- **Plan–Execute–Verify (PEV)** — a verification step after *each* executed step. *Wins:* error-prone tools, high-stakes steps; catches drift early. *Loses:* cost — verification per step roughly doubles calls.
- **Reflection** (agentic form) — generate → self-critique → refine within the loop. *Wins:* quality headroom on drafting tasks. *Loses:* self-review rationalizes its own errors (you'll fix this properly with a separate critic at Level 9).
- **Chain-of-Verification (CoVe)** — draft an answer, generate verification questions, answer them independently, revise. *Wins:* factual tasks; measurably cuts hallucination. *Loses:* creative/subjective tasks; adds fixed cost.
- **Self-Discover** — the model first selects and adapts reasoning strategies for the task, then applies them. *Wins:* diverse hard reasoning where no single strategy fits. *Loses:* routine tasks (strategy selection is pure overhead).
- **Constitutional AI (pattern form)** — a written rule list; outputs are checked per-rule and revised on failure. *Wins:* enforceable, explicit output constraints. *Loses:* rules that are vague — per-rule pass/fail forces you to write checkable rules, which is the actual value.

**Project menu (pick one):**
- [ ] Filesystem detective answering multi-step questions about a directory tree
- [ ] Research assistant that decomposes a question, gathers from pages, cites sources
- [ ] Log investigator: symptom in, root-cause hypothesis out
- [ ] Dataset validator: rules in, findings report out

**Done when:** ≥80% of a ten-task suite completes with correct termination in both directions, the loop detector fires and recovers unaided, and your ledger has honest first-pass entries for all seven patterns.

## Level 4 — Memory and RAG

**What it is.** Everything so far dies with the process. **Memory** is state that survives across runs; **RAG** (retrieval-augmented generation) is the standard mechanism — store text as **embeddings** (vectors capturing meaning) in a **vector database**, retrieve the most relevant pieces at question time, put them in the context. Memory flavors worth knowing by name: **episodic** (records of past runs), **semantic** (facts and knowledge), **procedural** (reusable skills).

**Why it's next.** Practically, most real use cases need it. Conceptually, it re-teaches Level 0 with stakes: memory only helps if the *right* piece lands in context at the right moment. A memory system is a context-management system — the decisions that matter are what to write, when to retrieve, and how many tokens of retrieved material to allow.

**Carry forward:** the agent loop, tools, logs.
**New this level:** vector store + embedding model; the retrieval pipeline (**chunking**, embedding, top-k similarity, keyword fallback, optional **reranking**); memory tools (`remember`, `recall`) so the model decides what's durable; episodic run-records retrieved at task start; a retrieval token budget; a procedural skill library.

**Basic Build — an agent that improves with reps.** Add all three memory flavors to your Level 3 agent. Run a family of similar tasks twenty times; chart performance across runs. Then two tests: (1) the **ablation** — disable memory, rerun, confirm the gain disappears; (2) the pathology hunt — find one real case each of: a similar-but-wrong memory retrieved confidently, trivia crowding out a decision, a stale memory applied after things changed. Fixing what you find forces the real design work: a write policy (storing everything is worse than nothing — retrieval noise misleads) and a read policy.

**Patterns to build here (ledger entries):**

*Retrieval shapes:*
- **Agentic RAG** — retrieval as a tool the agent calls deliberately, iterating queries. *Wins:* flexibility, token efficiency. *Loses:* the agent may not know it needs to retrieve.
- **Corrective RAG (CRAG)** — grade retrieved docs; on poor grades, fall back (e.g., to web search). *Wins:* spotty knowledge bases. *Loses:* the grader is another judge to calibrate.
- **Self-RAG** — the model itself emits decisions: retrieve or not, is this doc relevant, is my claim supported. *Wins:* mixed workloads where retrieval isn't always needed. *Loses:* complexity; more decisions to go wrong.
- **Adaptive RAG** — route by query complexity first: easy → no retrieval, medium → single-shot, hard → iterative. *Wins:* cost at scale. *Loses:* the router misjudging complexity (your Level 2 routing lesson again).
- **GraphRAG** — build a knowledge graph + community summaries from the corpus; answer from graph structure. *Wins:* "global" questions spanning the whole corpus that top-k chunks can't see. *Loses:* heavy indexing cost; overkill for lookup questions.

*Memory shapes:*
- **Episodic + Semantic memory** — turns + extracted fact triples. *Wins:* the sane default. *Loses:* nothing specific; it's the baseline shape.
- **Graph Memory** — (subject, predicate, object) triples with relations. *Wins:* relational questions ("who works with whom"). *Loses:* extraction quality gates everything.
- **MemGPT-style tiered context** — treat context like an OS treats RAM: a managed main context plus archival storage, with the model paging things in and out. *Wins:* very long-lived sessions. *Loses:* machinery overhead for short ones.
- **Voyager-style skill library** — verified, reusable code skills indexed by task similarity. *Wins:* compounding capability in a stable environment. *Loses:* skills rot when the environment shifts.
- **Agent Workflow Memory** — store *workflow recipes* (successful multi-step procedures) and retrieve them for similar tasks. *Wins:* recurring task families. *Loses:* recipes misapplied to near-miss tasks — your similar-but-wrong pathology, at procedure scale.

**Project menu (pick one):**
- [ ] Personal knowledge base you can talk to (notes, bookmarks, papers)
- [ ] Q&A assistant over one real codebase
- [ ] Docs bot for a tool you use, built from its manuals
- [ ] Learning coach that remembers what you studied and where you struggled

**Done when:** Run 20 beats run 1 measurably, the ablation kills the gain, your write policy and retrieval budget each fit in a sentence — and the ledger says, for each of the ten shapes above, what kind of question it's for.

## Level 5 — Code Execution and Sandboxing

**What it is.** The most powerful tool you can give an agent: writing and running code. It arrives with a twin obligation — a **sandbox**: an isolated container (Docker/Podman) with a limited filesystem, restricted network, resource caps, and timeouts, because sooner or later the model *will* write something destructive or infinite.

**Why it's next.** Beyond capability, this level teaches the deepest lesson in the book: **the environment's feedback beats the model's reasoning.** A traceback is free, perfectly accurate supervision — and an agent that reads the error, fixes, and retries (the **repair loop**) outperforms a smarter agent that only reasons harder. Rich, honest, fast feedback is what makes agents work; you'll spend the rest of the book seeking it.

**Carry forward:** the agent loop, termination machinery.
**New this level:** the sandbox; execution tools (`run_code`, `run_shell`, `read_file`, `write_file`) returning stdout/stderr/exit code; smart output truncation (head + tail + "N lines omitted" — how you cut 10,000 lines of stderr decides whether the model can debug); the repair loop with an attempt budget and a forced "try a different approach" after N identical failures.

**Basic Build — a self-healing executor.** "Here's a messy CSV — produce a cleaned version and a chart," end to end, no help, surviving three of its own errors. Then: a task where the first fix hypothesis is wrong (watch fixation; watch your intervention break it), and a deliberate destructive attempt to prove the sandbox holds. Classic pathology to catch: the agent "fixes" an error by deleting the code that raised it.

**Patterns to build here (ledger entries):**
- **SWE-Agent style** — the agent-computer interface idea: purpose-built tools for navigating and editing a real codebase (search, view-with-line-numbers, targeted edit) rather than raw shell. *Wins:* demonstrates the book's tool-design thesis — better interfaces beat better models; this is tool calling and sandboxing composed into the field's flagship application. *Loses:* interface design effort is per-domain.
- **Dry-Run** — propose the action, *simulate* or preview its effect, require an approval gate before executing anything irreversible. *Wins:* any tool with side effects; the pattern behind every "human-in-the-loop" production deployment. *Loses:* approval fatigue if you gate everything — tier your actions.

**Project menu (pick one):**
- [ ] Data-wrangling agent: messy files in, clean data and charts out
- [ ] Test fixer: repo with failing tests in, green tests out
- [ ] Analysis agent: question in, executed analysis with figures out
- [ ] Script medic: watches a recurring job, diagnoses and patches failures

**Done when:** 7 of 10 varied tasks complete unaided including multi-error recoveries, and the sandbox demonstrably contains a destructive command.

---

# Part II — Engineering Discipline (Levels 6–9)

## Level 6 — Evals

**What it is.** An **eval** is how you measure an agent: tasks, a definition of success per task, and a runner that executes everything repeatedly and reports pass rates. Repeatedly is the operative word — LLMs are stochastic; the same agent passes a task on one run and fails the next. Single runs tell you nothing; distributions tell the truth.

**Why it's the keystone.** From here on, every question that matters — did that change help? which pattern wins? did fine-tuning work? — is only answerable with evals. This machinery literally becomes: the verifier (Level 7), the trajectory filter (Level 10), the reward function (Level 11), the objective (Level 12), and the success detector (Level 13). One sentence to internalize: *whoever can check work automatically can improve work automatically.* And it's what turns your Pattern Ledger from opinions into evidence — after this level, you go back and attach numbers to every verdict you've written so far. Bluntly: this level separates people who demo agents from people who engineer them. Almost everyone skips it. Don't.

**Carry forward:** everything — evals wrap around your existing agents and patterns.
**New this level:** a task suite (20–50 tasks) with per-task success criteria; **programmatic checkers** wherever success is mechanical (file parses, tests pass, answer matches) — always preferred; **LLM-as-judge** (a model grading against a written rubric) only where mechanical checks are impossible, *calibrated first* against hand labels, because judges have measurable biases (longer answers, earlier options, their own model family); a runner doing N≥5 runs per configuration collecting pass rate, steps, tokens, cost; a results store with a comparison view (per-task deltas, with variance); a **held-out set** you never tune against; **tracing** — every model call logged with full prompt, response, and cost, every run replayable (a structured log file per run gets you most of what commercial observability tools sell).

**Basic Build — the harness that kills a belief.** Build the runner over your Levels 3–5 agents. Choose a change you're *confident* helps, write your predicted effect down, run N≥5 before and after, and look. Repeat until the data kills at least one belief. That experience — not the code — is the level. Then do the ledger pass: rerun your Level 2–5 patterns through the harness and upgrade every "wins/loses" entry from impression to measurement.

**This level is also called prompt testing.** In real teams, the single most common everyday use of an eval harness is answering "did this prompt change help?" — and the belief your harness kills in the basic build will very likely be a prompt belief. Two habits make it a discipline. First, the **prompt regression suite**: treat every prompt edit with the same ceremony as a code change — diff it, run the suite, compare against the last known-good numbers, then commit; never ship a prompt edit on vibes again. Second, **prompt management**: prompts live in version control as first-class artifacts — templated, versioned, with their eval results attached — not as strings buried in code. It's unglamorous, and it's exactly what separates teams who can answer "why did the agent get worse last Tuesday?" from teams who can't. (This habit pays compound interest at Level 9, where a long-running agent's prompts evolve across weeks, and it's a hard prerequisite for Level 12, where a machine starts editing your prompts and every change needs a diffable, revertible history.)

**Project menu (pick one):**
- [ ] Regression suite for your best agent, run on every change
- [ ] Model shootout: three models through one harness, quality-vs-cost report
- [ ] Judge calibration study: your LLM judge vs. your hand labels, biases quantified
- [ ] Pattern bakeoff #1: all your RAG shapes on one question set — the first measured page of the ledger

**Done when:** You can answer "did that change help?" with pass rates and variance — and you own one belief your own data killed.

## Level 7 — Test-Time Compute

**What it is.** **Test-time compute** (inference-time scaling): buying capability with tokens instead of training — sample several attempts, keep the best. Standard forms: **best-of-N** (a verifier picks the winner), **self-consistency** (majority-vote across sampled reasoning paths), retry-on-failure, and search over steps or actions (expand candidates, score, prune). This is the idea behind modern reasoning models, available to you as harness code.

**Why it's next.** It reframes the model: not an oracle asked once, but a **stochastic policy you sample from** — variance becomes raw material. It's the cheapest capability upgrade in the book (often 10–20 points on hard tasks, no training), and it's your first *consumer* of Level 6: best-of-N is only as good as the verifier picking winners. A sloppy verifier turns best-of-N into worst-of-N — you're actively selecting for whatever fools it.

**Carry forward:** everything, plus Level 6 checkers now used *during* execution.
**New this level:** the sampler (N diverse attempts via temperature/prompt variation); the aggregator (verifier selection; voting); retry policies; cost accounting — every result is now a point on an **accuracy-versus-cost curve**, never a single number.

**Basic Build — the accuracy/cost explorer.** One hard task family from your suite. Four strategies — single attempt, best-of-5 with verifier, self-consistency-of-5, retry-on-failure — at N≥5 runs each; plot accuracy against cost. Find the crossover where a *small* model plus sampling beats a *big* model single-shot (it exists, and finding it changes how you think about model choice). Then sabotage it: swap in a lax verifier and watch best-of-N select confident garbage.

**Patterns to build here (ledger entries):**
- **Self-Consistency** — sample N reasoning paths, majority-vote the answer. *Wins:* problems with a single extractable answer; embarrassingly simple. *Loses:* open-ended outputs (nothing to vote on); groupthink when the model has a systematic bias.
- **Ensemble** — N differently-prompted (or differently-modeled) voters, weighted aggregation. *Wins:* decorrelated errors. *Loses:* correlated voters vote wrong together — diversity is the whole trick.
- **Tree of Thoughts (ToT)** — grow partial solutions as a tree; score and expand the promising ones (beam search over thoughts). *Wins:* problems needing exploration and backtracking (puzzles, planning). *Loses:* linear tasks — the tree is pure overhead.
- **LATS** — MCTS over agent actions: simulate, score, back up rewards, expand the best branches. *Wins:* action spaces where lookahead pays and a decent value signal exists. *Loses:* cost explodes; the published benchmark result to remember — it *underperforms on simple tasks*; wrong shape for the job is a real failure mode.
- **Mental Loop / simulate-then-act** — before acting, simulate the action's outcome and score it deterministically; act only on the best. *Wins:* expensive or risky actions, cheap simulations. *Loses:* simulation fidelity — a wrong world model confidently picks wrong actions.

**Project menu (pick one):**
- [ ] Hard-problem solver where single attempts fail and voting wins
- [ ] Draft-and-select writer: N drafts, rubric verifier, automatic winner
- [ ] Your Level 5 executor with retry-on-failure, before/after measured
- [ ] Small-model equalizer: how close does a small local model get to a frontier model, given samples?

**Done when:** The accuracy/cost curve exists for ≥3 strategies on one task family, you can argue where production should sit on it, and the ledger records the task shapes where ToT/LATS pay and where they burn money.

## Level 8 — Multi-Agent Systems

**What it is.** Several agents with roles working one task: typically an **orchestrator** decomposing and dispatching to **specialists** — each with its own prompt, tools, and (the actual point) its own *clean context* — coordinating through structured handoffs and shared state rather than one giant conversation.

**Why it's next — and why it's a trap.** The most hyped pattern in the field, and the hype inverts the truth: multi-agent is a *cost* paid for specific benefits. Every handoff loses information; errors compound; agents duplicate and undo each other's work; tokens multiply. The real benefits: context isolation, tool scoping, parallelism over independent subtasks, and genuinely **adversarial review** (a critic in a separate context catches what self-review rationalizes — but it needs structural separation and read-only access; "be critical" in a prompt produces rubber-stamping). This level replaces the hype with data: on many tasks one good agent beats five mediocre ones, and you should know from your own evals which of your tasks are which.

**Carry forward:** everything; specialists are configured instances of your existing agent; orchestrator–workers from Level 2 is the workflow ancestor of all of this.
**New this level:** the orchestrator (also your bottleneck and single point of failure); specialist configs; **handoff schemas** (structured task-in/result-out, never raw transcripts — and written with tool-description care, because a specialist solves what the spec *says*); shared state (task board or files); the critic role.

**Basic Build — the honest comparison.** Orchestrator + researcher + executor + critic for one capability; the *same* capability as one well-prompted agent with all the tools; both through your harness, same tasks, same model, N≥5, reporting quality *and* cost. Deliverable: a paragraph stating which task properties favored decomposition in your data. Expect at least one uncomfortable result.

**Patterns to build here (ledger entries):**
- **Supervisor / Multi-Agent** — the orchestrator-specialists shape above. *Wins:* separable subtasks needing different tools/contexts. *Loses:* tightly-coupled tasks; handoff loss dominates.
- **Blackboard** — no dispatcher: agents watch a shared workspace and contribute when they can, until a solution assembles. *Wins:* problems where contribution order is unknowable. *Loses:* no one is responsible for finishing — termination gets *harder*.
- **Debate** — N agents argue opposing positions for K rounds; a judge or vote decides. *Wins:* questions with genuine tension where positions sharpen each other. *Loses:* the published failure to remember — group-think on trick questions: agents converge confidently on the same wrong answer. Debate is not a truth machine.
- **STORM** — multi-perspective research: generate distinct personas, each interviews/researches from its angle, synthesize into a structured article. *Wins:* broad survey-style outputs where coverage matters. *Loses:* depth on narrow questions; heavy cost.
- **Meta-Controller** — a router *over architectures*: classify the task, dispatch to the best pattern in your zoo. *Wins:* exactly proportional to how good your ledger is — this pattern is your capstone's engine. *Loses:* misclassification; a meta-controller is only as good as its map of when each pattern wins.

**Project menu (pick one):**
- [ ] **The pattern bakeoff (recommended):** supervisor vs. blackboard vs. debate vs. STORM vs. your single-agent baseline, one interface, one task suite, one leaderboard — the ledger's centerpiece page
- [ ] Research pipeline (gatherer → synthesizer → fact-check critic) vs. one generalist
- [ ] Builder + adversarial reviewer vs. builder with self-review
- [ ] Two advocates + judge on contested questions vs. one balanced agent

**Done when:** You hold head-to-head data across the multi-agent shapes and can state from evidence — not vibes — when each pays for itself.

## Level 9 — Long-Running Agents

**What it is.** Agents operating for hours or days on standing tasks — services, not commands. The engineering is reliability: **checkpointing** (save full state every iteration; crash → clean resume), **idempotency** (a resumed step never double-applies a side effect — never send the email twice), budget caps (tokens, dollars, steps; degrade gracefully, don't die mid-action), **escalation** (a defined "I'm stuck" state that pauses and asks a human with enough context to answer in one message), an append-only **audit log**, and drift control (periodically re-read the original goal; 200 steps in, agents wander).

**Why it's next.** Failure becomes statistical at this horizon: 1% per step is invisible over 10 steps and near-certain over 500. The design center shifts from prevention to *recovery* — and you meet the field's open secret: in a long-running system the model is maybe 20% of the engineering. Two calibrations you can only learn by doing: escalation (never asking = silent failure; always asking = not autonomous), and compaction — your Level 0 exercise, now production infrastructure, with the new failure mode of **slow context poisoning**: one bad summary compounding quietly for days.

**Carry forward:** everything; this level hardens it.
**New this level:** checkpoint/restore, idempotency keys, the budget governor, the escalation handler, the audit log, drift re-grounding, a scheduler (cron/event triggers).

**Basic Build — the survivor.** A standing-task agent, 24+ hours, on something real. During the run: `kill -9` it twice at bad moments, verify clean resume with zero duplicated side effects; force one escalation and grade its ask; afterward, reconstruct one action's full "why" from the audit log alone.

**Patterns to build here (ledger entries):**
- **Reflexion** — after each failed episode, the agent writes a verbal reflection ("what went wrong, what to try") into episodic memory, retrieved on retry. *Wins:* repeated attempts at the same task family; learning across episodes without training. *Loses:* the published failure — wrong memory shape for raw fact recall; reflections are lessons, not a database.
- **Reflexive Metacognitive** — the agent maintains a model of its *own* capabilities and routes accordingly: attempt, delegate, or escalate. *Wins:* the escalation-calibration problem, made explicit — the pattern behind a well-tuned "I'm stuck" state. *Loses:* self-assessment is exactly what LLMs are worst at; ground it in your eval data, not the model's confidence.

**Project menu (pick one):**
- [ ] Repo steward: triages issues, labels, drafts replies, escalates judgment calls
- [ ] Daily brief: monitors sources on a topic, compiles every morning
- [ ] Pipeline warden: watches a recurring job, fixes known failure classes, escalates novel ones
- [ ] Event-stream analyst: continuously summarizes and prioritizes a log/alert stream

**Done when:** 24+ hours, two kills, zero duplicate side effects, one clean escalation, any action explainable from the log.

---

# Part III — Training the Model Itself (Levels 10–13)

*Until now the model was fixed. Now you change the weights. Everything here runs on Level 6's machinery. A small open model (3–8B) and one GPU suffice throughout.*

## Level 10 — Fine-Tuning on Agent Trajectories

**What it is.** **Fine-tuning** continues a model's training on your data. For agents, the data is **trajectories** — complete recorded runs (context, tool calls, results, decisions) filtered to successes, then trained in. Result: a model that *natively* knows your tools, formats, and termination behavior. Sub-skills: **LoRA** (cheap adapter-weight tuning — a weekend on one GPU with Unsloth or Axolotl) and **distillation** — frontier-model trajectories trained into a small local model: capability transferred downmarket.

**Why it's next.** It closes the loop between harness and model: your Level 6 criteria decide which trajectories are learn-worthy, so *your evals' definition of good — blind spots included — gets baked into weights.* Curation is where the leverage lives; training is the easy part. And it teaches an honest trade: reliability on your distribution, paid for with generality — so the level requires measuring what you lost, not just what you gained.

**Carry forward:** the harness (recorder and judge), the agent stack (demonstration generator).
**New this level:** trajectory capture in training format; the success filter; a dataset formatter; the LoRA trainer; a model registry (every checkpoint stored with its data and eval results).

**Basic Build — the harness-taught model.** A few hundred trajectories from your best agent → filter to successes → LoRA-tune a small model → evaluate *in the same harness* against its base on held-out tasks. Then distill: frontier trajectories on the same tasks into the small model; measure the transfer. Finally, hunt the cost: one failure the tuned model has that the base didn't.

**Project menu (pick one):**
- [ ] House specialist: a local model tuned to your exact tools, benchmarked against base
- [ ] Frontier distillate: big-model behavior compressed into a small model
- [ ] Format enforcer: near-zero structured-output violations, trained in
- [ ] Termination tuner: trajectories filtered for perfect finishing, trained in

**Done when:** The tuned model beats base in-harness on held-out tasks — and you can exhibit one failure it *gained*.

## Level 11 — Reinforcement Learning from Verifiable Rewards

**What it is.** **RLVR** — the technique behind modern reasoning models. Instead of imitating (Level 10), the model *generates* many attempts, an automatic **verifier** scores them (tests pass? answer correct?), and **reinforcement learning** shifts the weights toward what scored. **GRPO** is the workhorse algorithm; open libraries (TRL, verl) run it on small models on one GPU. The prerequisite insight: your Level 6 programmatic checkers *are* reward functions — RLVR is evals with the stakes turned up.

**Why it's next.** It teaches, viscerally, applied ML's most important lesson: **the model learns exactly what you measure, not what you meant.** A reward with a seam — hardcode-able tests, gameable string matches — *will* be exploited; that's not a risk but the default outcome (reward hacking; Goodhart's law with a gradient). Building verifiers that survive an optimizer probing them is a discipline learnable only by catching your own model cheating. One toy run permanently changes how you read model behavior: every model becomes the residue of its training signal.

**Carry forward:** the sandbox (verification substrate), checkers (now rewards), the harness (before/after).
**New this level:** the reward function as training signal; a rollout generator; the RL trainer; reward-curve instrumentation *plus manual rollout inspection* (reward up ≠ the capability you wanted up); in parallel, one trained **reward model** — outcome-scoring vs. process-scoring (grade the answer vs. grade each step).

**Basic Build — the Goodhart safari.** One small RLVR run end to end on a mechanically verifiable domain (short programming tasks with generated tests, in your sandbox). Watch the reward climb — then audit rollouts by hand until you find the policy exploiting or probing a seam (hardcoded test values, gamed matches — whatever your checker permits). Document the hack. Patch the verifier. Run again. Finding the exploit *yourself* is the level.

**Patterns to build here (ledger entries):**
- **RLHF-style self-improvement (pattern form)** — multi-dimensional deterministic scoring over generations, archive the best, learn from the archive. *Wins:* the harness-side rehearsal of this level's ideas — no gradients needed to feel the dynamics. *Loses:* everything a gameable score loses, which is the point.

**Project menu (pick one):**
- [ ] Verified-code specialist: RLVR on programming puzzles, held-out before/after
- [ ] Math sharpener: a small model against a symbolic checker
- [ ] Reward-model bakeoff: outcome vs. process scoring, disagreements analyzed
- [ ] Standing red-team of your own verifiers: a documented exploit hunt per checker

**Done when:** One completed run with curves, a held-out before/after — and a documented reward exploit you found on purpose.

## Level 12 — Automated Optimization

**What it is.** Closing the outermost loop: a system that improves your agent automatically — proposing changes to prompts, tool descriptions, retrieval settings, sampling strategies (ambitiously: triggering Level 10 retrains), testing each against your evals, keeping only measured wins. The prompt-specific version has mature tooling (DSPy, GEPA: prompts as parameters, evals as objective); the general version is an optimizer agent reading failure traces.

**Why it's next.** It's Level 6's logic concluded — measure automatically, improve automatically — with the last calibration lesson: **the loop amplifies exactly what your evals measure, blind spots included.** The guardrails are the engineering: one bounded change at a time; a statistical gate (+2% at N=5 is noise, and the optimizer must know it); a **frozen holdout** the optimizer never sees (overfitting the visible suite is otherwise guaranteed); immutable constraints; version control on everything. Most proposals die on re-evaluation — the value is the *ratchet*: gains accumulate, regressions get caught.

**Carry forward:** everything; the optimizer sits on top.
**New this level:** the optimizer process, the one-change-per-branch protocol, the statistical keep/revert gate, the frozen holdout, the constraint file.

**Basic Build — the ratchet.** Point the optimizer at one agent for a fixed budget, across two change types: automated prompt optimization, and one non-prompt change (a tool description, a retrieval budget, a sampling strategy). Deliverable: one surviving improvement *you did not design*, holding on the frozen holdout, full propose→test→keep/revert history in git.

**Project menu (pick one):**
- [ ] Prompt evolver aimed at your worst-performing agent
- [ ] Tool-description optimizer scored on tool-selection accuracy
- [ ] Config search over compaction thresholds, retrieval budgets, sampling settings
- [ ] Full ratchet: prompts *and* Level 10 retrains under the same gates

**Done when:** The loop produced an improvement you didn't design, it survives the holdout, and the history is in version control.

## Level 13 — Open-Ended Agents

**What it is.** The research frontier: a system generating its *own* curriculum — proposing tasks just beyond its ability, attempting them, verifying successes, archiving skills, escalating difficulty, indefinitely. Reference points: **Voyager** (self-taught Minecraft skills, kept in a library — you built its memory shape at Level 4) and **POET** (co-evolving tasks and solvers). It's a teacher/student architecture where you build the teacher: a **task generator** (conditioned on current skills — tasks land *solvable-but-not-solved*), a **success-detector generator** (novel tasks can't have hand-written graders), an **append-mostly skill archive** (keep odd, currently-useless solutions — progress routes through stepping stones), and a **novelty filter** (new-enough vs. learnable-enough: pure novelty yields garbage, pure repetition yields nothing).

**Why it's last.** It's the whole book recomposed: the solver is your existing agent unchanged (needing *nothing new* is the lesson); the detector generator is Level 6 under the hardest conditions — gameable means the system optimizes toward gaming it (Level 11 at system scale); the outer loop needs all of Level 9's governance, because open-ended systems burn tokens by design. The field's hard-won debugging heuristic: when it stagnates, it's almost always the *teacher* — generator overshooting or filter miscalibrated — not a weak solver.

**Patterns to build here (ledger entries):**
- **Voyager (full loop)** — automatic curriculum + skill library + iterative refinement, in an environment. *Wins:* stable environments with verifiable success. *Loses:* environment shift rots the library.
- **Cellular Automata (LLM rules over a grid)** — the oddball of the 35: local LLM-decided rules, global emergent behavior. *Wins:* as a cheap laboratory for emergence — the smallest system that surprises you. *Loses:* it's a toy; its ledger entry is about what emergence *feels* like to debug.

**Basic Build — the self-curriculum.** Assemble the teacher over programming tasks in your sandbox (generated-test verification is most trustworthy there). Run for days under budget caps; chart the archive's difficulty frontier over time. Expect and treat the three diseases: curriculum collapse (trivial variants forever), difficulty cliffs (nothing solves for days), archive pollution (false-positive successes corrupting later attempts).

**Project menu (pick one):**
- [ ] Coding-skill grower: self-generated programming curriculum with a visible frontier
- [ ] Tool inventor: finds its own capability gaps, writes-and-verifies new tools
- [ ] Game-skill explorer: self-curriculum in a scriptable game or simulator
- [ ] Domain explorer: self-generated question/verification cycles over a knowledge field

**Done when:** There's no clean criterion — that's the point. The soft one: the difficulty frontier visibly advances over days of autonomous operation, and the system solves something you never specified. When it surprises you usefully, you've crossed from studying the field to contributing to it.

---

# Elective — Computer Use

**What it is.** Agents that operate software the way a person does: screenshot → vision-capable model → click/type → screenshot → verify. It exists because enormous amounts of software — nearly all legacy and industrial software — have no API; the screen is the interface. The full stack adds **grounding** (mapping "the Submit button" to coordinates — the flakiest part), physical actuation, and mandatory per-step verification, plus an explicit **action-risk policy** (reads free; mutations gated; irreversible actions forbidden), because perception is now probabilistic and a click has no return value.

**Why it's an elective, not a level.** The *concepts* it teaches — verification loops, graduated autonomy — you'll already have from Dry-Run (Level 5) and escalation design (Level 9). What remains is a large, flaky, domain-specific engineering lift, worth doing when you have a concrete target that has no API — not as a rung everyone must climb. Meanwhile the reliable half needs no elective at all: read-only vision ("screenshot or photo in, structured values out") is just Level 1 tool calling with an image, and it's listed there as a project option.

**If you take it:** complete one multi-step task through a purely visual interface with per-step verification; survive a mid-task layout change; write the action-risk policy with one line of justification per tier. Ledger entry: **BrowserAgent** — *wins:* software with no API; *loses:* reliability — an order of magnitude below API tools, which is why reads ship long before writes.

---




