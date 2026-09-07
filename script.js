/**
 * MEMORY LAB — Understanding Linear Attention
 * Core Interactive & Scientific Substrate (Upgraded Engine)
 * 
 * Features:
 * 1. Animated Hero Memory Engine (Continuous traveling token pipeline)
 * 2. Pairwise Footprint Visualizer (Exact JS calculation for N in {4, 8, 12, 16})
 * 3. 100-Token Race (Dynamic sequence slider 4 to 100 with Canvas/1D stream growth animation)
 * 4. Toy Linear Attention Engine (4D ELU+1 feature map, outer-product updates, state accumulation)
 * 5. Simplified Default Memory Flow with Collapsible Detailed Math
 * 6. "Truth Beside Estimate" Prediction Challenge (Updated for 4-100 range)
 * 7. Multi-Tab BDH / BDH-CQ Research Connection
 * 8. Scroll Section Tracking & Progress Rail
 */

(function() {
    'use strict';

    // =========================================================================
    // 1. DATASETS & TOY FEATURE DEFINITIONS (d = 4)
    // =========================================================================
    // 4 Dimensions: [Entity/Subject, Action/Relation, Attribute, Value/Classification]
    const EXPERIMENT_SCENARIOS = {
        associative: {
            title: "Associative Key-Value Recall",
            tokens: [
                { word: "apple",   k: [0.85, -0.10, 0.40, 0.15], v: [0.95, 0.05, 0.10, 0.00], category: "Fruit Entity" },
                { word: "is",      k: [-0.05, 0.80, 0.10, 0.05], v: [0.10, 0.80, 0.10, 0.00], category: "Copula Relation" },
                { word: "fruit",   k: [0.80, 0.05, 0.35, 0.10], v: [0.90, 0.15, 0.05, 0.00], category: "Class Target" },
                { word: "wolf",    k: [-0.20, 0.15, 0.85, 0.20], v: [0.05, 0.10, 0.92, 0.00], category: "Fauna Entity" },
                { word: "is",      k: [-0.05, 0.80, 0.10, 0.05], v: [0.10, 0.80, 0.10, 0.00], category: "Copula Relation" },
                { word: "animal",  k: [-0.15, 0.10, 0.80, 0.15], v: [0.05, 0.15, 0.88, 0.00], category: "Class Target" },
                { word: "ruby",    k: [0.10, -0.20, 0.15, 0.90], v: [0.00, 0.05, 0.10, 0.95], category: "Mineral Entity" },
                { word: "is",      k: [-0.05, 0.80, 0.10, 0.05], v: [0.10, 0.80, 0.10, 0.00], category: "Copula Relation" },
                { word: "gem",     k: [0.15, -0.15, 0.10, 0.85], v: [0.00, 0.10, 0.05, 0.92], category: "Class Target" }
            ]
        },
        sentiment: {
            title: "Sequential Sentiment Drift",
            tokens: [
                { word: "The",       k: [0.10, 0.10, 0.10, 0.10], v: [0.05, 0.05, 0.05, 0.05], category: "Determiner" },
                { word: "software",  k: [0.80, 0.20, 0.10, 0.00], v: [0.60, 0.10, 0.20, 0.00], category: "Subject" },
                { word: "was",       k: [0.00, 0.70, 0.00, 0.00], v: [0.10, 0.60, 0.00, 0.00], category: "Tense" },
                { word: "flawless",  k: [0.10, 0.20, 0.90, -0.10], v: [0.10, 0.00, 0.95, 0.00], category: "Positive Sentiment" },
                { word: "until",     k: [-0.10, 0.80, -0.20, 0.40], v: [0.00, 0.30, -0.20, 0.50], category: "Pivot" },
                { word: "sudden",    k: [0.20, 0.40, -0.40, 0.80], v: [0.10, 0.20, -0.30, 0.75], category: "Modifier" },
                { word: "failure",   k: [-0.10, 0.10, -0.85, 0.90], v: [0.00, 0.00, -0.90, 0.95], category: "Negative Shock" }
            ]
        },
        syntactic: {
            title: "Grammar S-V-O Agreement",
            tokens: [
                { word: "The",         k: [0.10, 0.10, 0.10, 0.10], v: [0.05, 0.05, 0.05, 0.05], category: "Determiner" },
                { word: "scientist",   k: [0.90, 0.10, 0.20, 0.00], v: [0.85, 0.15, 0.10, 0.00], category: "Agent (Singular)" },
                { word: "validates",   k: [0.30, 0.90, 0.10, 0.00], v: [0.20, 0.88, 0.10, 0.00], category: "Verb (Singular)" },
                { word: "the",         k: [0.10, 0.10, 0.10, 0.10], v: [0.05, 0.05, 0.05, 0.05], category: "Determiner" },
                { word: "hypothesis",  k: [0.20, 0.30, 0.90, 0.10], v: [0.15, 0.20, 0.90, 0.00], category: "Patient / Object" },
                { word: "rigorously",  k: [0.10, 0.40, 0.30, 0.85], v: [0.10, 0.35, 0.20, 0.80], category: "Adverbial Manner" }
            ]
        }
    };

    // Mathematical Feature Map: ELU(x) + 1 (Ensures non-negative kernel values >= 0)
    function featureMapELU(vec) {
        return vec.map(x => (x > 0 ? x + 1.0 : Math.exp(x)));
    }

    // Outer Product: u \otimes v^T -> d x d Matrix
    function outerProduct(u, v) {
        const dimU = u.length;
        const dimV = v.length;
        const mat = [];
        for (let i = 0; i < dimU; i++) {
            mat[i] = [];
            for (let j = 0; j < dimV; j++) {
                mat[i][j] = u[i] * v[j];
            }
        }
        return mat;
    }

    // Matrix Addition: A + B
    function addMatrices(A, B) {
        const rows = A.length;
        const cols = A[0].length;
        const res = [];
        for (let i = 0; i < rows; i++) {
            res[i] = [];
            for (let j = 0; j < cols; j++) {
                res[i][j] = A[i][j] + B[i][j];
            }
        }
        return res;
    }

    // Frobenius Norm: ||M||_F
    function frobeniusNorm(mat) {
        let sumSq = 0;
        for (let i = 0; i < mat.length; i++) {
            for (let j = 0; j < mat[i].length; j++) {
                sumSq += mat[i][j] * mat[i][j];
            }
        }
        return Math.sqrt(sumSq);
    }

    // Vector Dot Product: u^T v
    function dotProduct(u, v) {
        let sum = 0;
        for (let i = 0; i < u.length; i++) {
            sum += u[i] * v[i];
        }
        return sum;
    }

    // Vector-Matrix Product: u^T M -> Row Vector
    function vecMatProduct(u, M) {
        const cols = M[0].length;
        const res = new Array(cols).fill(0);
        for (let j = 0; j < cols; j++) {
            for (let i = 0; i < u.length; i++) {
                res[j] += u[i] * M[i][j];
            }
        }
        return res;
    }

    // Create Zero Matrix of size rows x cols
    function createZeroMatrix(rows, cols) {
        const m = [];
        for (let i = 0; i < rows; i++) {
            m[i] = new Array(cols).fill(0);
        }
        return m;
    }


    // =========================================================================
    // 2. HERO MEMORY ENGINE (Continuous Traveling Token Animation)
    // =========================================================================
    // =========================================================================
    // 2. HERO MEMORY ENGINE (Deterministic 8-Token Sequential Pipeline)
    // =========================================================================
    const HERO_TOKENS = [
        { word: "The",        k: [0.10, 0.10, 0.10, 0.10], v: [0.05, 0.05, 0.05, 0.05] },
        { word: "recurrent",  k: [0.20, 0.85, 0.15, 0.10], v: [0.15, 0.90, 0.10, 0.05] },
        { word: "state",      k: [0.85, 0.20, 0.30, 0.10], v: [0.80, 0.25, 0.20, 0.10] },
        { word: "updates",    k: [0.15, 0.90, 0.20, 0.15], v: [0.10, 0.85, 0.25, 0.10] },
        { word: "linearly",   k: [0.30, 0.40, 0.85, 0.10], v: [0.25, 0.35, 0.90, 0.05] },
        { word: "without",    k: [-0.10, 0.50, -0.20, 0.40], v: [0.00, 0.40, -0.10, 0.50] },
        { word: "pairwise",   k: [0.70, -0.30, 0.60, 0.20], v: [0.65, -0.20, 0.70, 0.15] },
        { word: "comparison", k: [0.40, 0.30, 0.50, 0.80], v: [0.35, 0.25, 0.45, 0.85] }
    ];

    let heroTokenIndex = 0;
    let heroIsPlaying = true;
    let heroCycleTimer = null;
    let heroPhaseTimeouts = [];
    let heroState = createZeroMatrix(3, 3);
    let heroNormalizer = [0, 0, 0];

    function clearHeroTimeouts() {
        heroPhaseTimeouts.forEach(id => clearTimeout(id));
        heroPhaseTimeouts = [];
    }

    function initHeroEngine() {
        const streamContainer = document.getElementById("heroTokenStream");
        const featureMini = document.getElementById("heroFeatureMini");
        const updateMini = document.getElementById("heroUpdateMini");
        const stateMini = document.getElementById("heroStateMini");

        const playBtn = document.getElementById("heroPlayBtn");
        const pauseBtn = document.getElementById("heroPauseBtn");
        const replayBtn = document.getElementById("heroReplayBtn");

        if (!streamContainer || !updateMini || !stateMini) return;

        // Render Initial 8 Token Badges
        streamContainer.innerHTML = "";
        HERO_TOKENS.forEach((item, idx) => {
            const pill = document.createElement("span");
            pill.className = `token-pill-mini ${idx === 0 ? 'active' : ''}`;
            pill.id = `hero-tok-${idx}`;
            pill.textContent = item.word;
            streamContainer.appendChild(pill);
        });

        // Initialize 4 feature mini bars
        if (featureMini) {
            featureMini.innerHTML = "";
            for (let i = 0; i < 4; i++) {
                const bar = document.createElement("div");
                bar.className = "mini-bar-unit";
                bar.id = `hero-bar-${i}`;
                bar.style.height = "16px";
                featureMini.appendChild(bar);
            }
        }

        // Initialize 3x3 cells for delta & state
        updateMini.innerHTML = "";
        stateMini.innerHTML = "";
        for (let i = 0; i < 9; i++) {
            const uCell = document.createElement("div");
            uCell.className = "mini-cell";
            uCell.id = `hero-u-${i}`;
            updateMini.appendChild(uCell);

            const sCell = document.createElement("div");
            sCell.className = "mini-cell";
            sCell.id = `hero-s-${i}`;
            stateMini.appendChild(sCell);
        }

        // Event listeners for Play, Pause, Replay
        if (playBtn) {
            playBtn.addEventListener("click", () => {
                if (!heroIsPlaying) {
                    if (heroTokenIndex >= HERO_TOKENS.length) {
                        replayHeroEngine();
                    } else {
                        playHeroEngine();
                    }
                }
            });
        }

        if (pauseBtn) {
            pauseBtn.addEventListener("click", () => {
                pauseHeroEngine();
            });
        }

        if (replayBtn) {
            replayBtn.addEventListener("click", () => {
                replayHeroEngine();
            });
        }

        // Start playing from token 0
        heroTokenIndex = 0;
        playHeroEngine();
    }

    function playHeroEngine() {
        heroIsPlaying = true;
        updateHeroButtonStates();

        if (heroCycleTimer) clearInterval(heroCycleTimer);
        runDeterministicHeroCycle();
        heroCycleTimer = setInterval(() => {
            if (heroTokenIndex < HERO_TOKENS.length) {
                runDeterministicHeroCycle();
            } else {
                finishHeroSequence();
            }
        }, 2500);
    }

    function pauseHeroEngine() {
        heroIsPlaying = false;
        if (heroCycleTimer) {
            clearInterval(heroCycleTimer);
            heroCycleTimer = null;
        }
        clearHeroTimeouts();
        updateHeroButtonStates();
    }

    function replayHeroEngine() {
        pauseHeroEngine();
        heroTokenIndex = 0;
        heroState = createZeroMatrix(3, 3);
        heroNormalizer = [0, 0, 0];

        // Reset mini cells display
        for (let i = 0; i < 9; i++) {
            const uCell = document.getElementById(`hero-u-${i}`);
            if (uCell) uCell.style.background = "var(--surface-3)";
            const sCell = document.getElementById(`hero-s-${i}`);
            if (sCell) sCell.style.background = "var(--surface-3)";
        }

        const replayBtn = document.getElementById("heroReplayBtn");
        if (replayBtn) replayBtn.style.display = "none";

        playHeroEngine();
    }

    function finishHeroSequence() {
        pauseHeroEngine();
        const statusEl = document.getElementById("heroEngineStatus");
        if (statusEl) {
            statusEl.textContent = "SEQUENCE COMPLETE — All 8 tokens accumulated incrementally into S_8";
        }
        const replayBtn = document.getElementById("heroReplayBtn");
        if (replayBtn) {
            replayBtn.style.display = "inline-flex";
            replayBtn.classList.add("active");
        }
        const playBtn = document.getElementById("heroPlayBtn");
        const pauseBtn = document.getElementById("heroPauseBtn");
        if (playBtn) playBtn.classList.remove("active");
        if (pauseBtn) pauseBtn.classList.remove("active");
    }

    function updateHeroButtonStates() {
        const playBtn = document.getElementById("heroPlayBtn");
        const pauseBtn = document.getElementById("heroPauseBtn");
        if (playBtn) playBtn.classList.toggle("active", heroIsPlaying);
        if (pauseBtn) pauseBtn.classList.toggle("active", !heroIsPlaying);
    }

    function runDeterministicHeroCycle() {
        if (heroTokenIndex >= HERO_TOKENS.length) {
            finishHeroSequence();
            return;
        }

        const tokData = HERO_TOKENS[heroTokenIndex];
        const currentTok = tokData.word;
        const currentK = tokData.k;
        const currentV = tokData.v;

        const counterEl = document.getElementById("heroTokenCounter");
        const statusEl = document.getElementById("heroEngineStatus");
        const travelerHead = document.getElementById("heroTravelerHead");
        const travelerLabel = document.getElementById("heroTravelerLabel");

        if (travelerLabel) travelerLabel.textContent = currentTok;
        if (counterEl) counterEl.textContent = `${heroTokenIndex + 1} / 8`;

        const blocks = [
            document.getElementById("pipeInputBlock"),
            document.getElementById("pipeTransformBlock"),
            document.getElementById("pipeUpdateBlock"),
            document.getElementById("pipeStateBlock"),
            document.getElementById("pipeQueryBlock")
        ];

        const conns = [
            document.getElementById("pipeConn1"),
            document.getElementById("pipeConn2"),
            document.getElementById("pipeConn3"),
            document.getElementById("pipeConn4")
        ];

        const railPositions = [8, 28, 50, 72, 92];

        clearHeroTimeouts();

        let phiValues = [];
        let deltaMat = createZeroMatrix(3, 3);

        // --- PHASE 1: INPUT x_t (0ms) ---
        heroPhaseTimeouts.push(setTimeout(() => {
            if (travelerHead) travelerHead.style.left = `${railPositions[0]}%`;
            blocks.forEach((b, i) => b && b.classList.toggle("active-stage", i === 0));
            conns.forEach(c => c && c.classList.remove("flow-active"));

            // Highlight token stream
            HERO_TOKENS.forEach((_, idx) => {
                const el = document.getElementById(`hero-tok-${idx}`);
                if (el) el.classList.toggle("active", idx === heroTokenIndex);
            });

            if (statusEl) {
                statusEl.textContent = `t = 0${heroTokenIndex + 1} | Token "${currentTok}" (x_${heroTokenIndex + 1}) enters stream`;
            }
        }, 0));

        // --- PHASE 2: KERNEL TRANSFORM φ(k_t) (450ms) ---
        heroPhaseTimeouts.push(setTimeout(() => {
            if (travelerHead) travelerHead.style.left = `${railPositions[1]}%`;
            blocks.forEach((b, i) => b && b.classList.toggle("active-stage", i === 1));
            if (conns[0]) conns[0].classList.add("flow-active");

            // Evaluate non-negative ELU+1 feature map
            phiValues = featureMapELU(currentK);

            // Animate feature bars deterministically
            for (let i = 0; i < 4; i++) {
                const bar = document.getElementById(`hero-bar-${i}`);
                if (bar) {
                    const h = Math.round(10 + Math.min(28, phiValues[i] * 12));
                    bar.style.height = `${h}px`;
                }
            }

            if (statusEl) {
                statusEl.textContent = `t = 0${heroTokenIndex + 1} | Kernel φ(k_${heroTokenIndex + 1}) evaluated (non-negative >= 0)`;
            }
        }, 450));

        // --- PHASE 3: MEMORY UPDATE ΔS_t (900ms) ---
        heroPhaseTimeouts.push(setTimeout(() => {
            if (travelerHead) travelerHead.style.left = `${railPositions[2]}%`;
            blocks.forEach((b, i) => b && b.classList.toggle("active-stage", i === 2));
            if (conns[1]) conns[1].classList.add("flow-active");

            // Compute rank-1 outer product (first 3 dims for 3x3 mini view)
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    const val = phiValues[i] * currentV[j];
                    deltaMat[i][j] = val;
                    const uCell = document.getElementById(`hero-u-${i * 3 + j}`);
                    if (uCell) {
                        const alpha = Math.min(1, Math.max(0.2, val * 1.2));
                        uCell.style.background = `rgba(195, 183, 255, ${alpha})`;
                    }
                }
            }

            if (statusEl) {
                statusEl.textContent = `t = 0${heroTokenIndex + 1} | Outer product ΔS_${heroTokenIndex + 1} = φ(k_${heroTokenIndex + 1}) ⊗ v_${heroTokenIndex + 1}^T formed`;
            }
        }, 900));

        // --- PHASE 4: CURRENT STATE S_t ACCUMULATION (1350ms) ---
        heroPhaseTimeouts.push(setTimeout(() => {
            if (travelerHead) travelerHead.style.left = `${railPositions[3]}%`;
            blocks.forEach((b, i) => b && b.classList.toggle("active-stage", i === 3));
            if (conns[2]) conns[2].classList.add("flow-active");

            // In-place accumulation into recurrent state S_t = S_{t-1} + ΔS_t
            for (let i = 0; i < 3; i++) {
                heroNormalizer[i] += phiValues[i];
                for (let j = 0; j < 3; j++) {
                    heroState[i][j] += deltaMat[i][j];
                    const sCell = document.getElementById(`hero-s-${i * 3 + j}`);
                    if (sCell) {
                        const alpha = Math.min(1, Math.max(0.25, heroState[i][j] / 2.0));
                        sCell.style.background = `rgba(169, 154, 244, ${alpha})`;
                    }
                }
            }

            if (statusEl) {
                statusEl.textContent = `t = 0${heroTokenIndex + 1} | S_${heroTokenIndex + 1} = S_${heroTokenIndex} + ΔS_${heroTokenIndex + 1} accumulated in-place`;
            }
        }, 1350));

        // --- PHASE 5: INSTANT QUERY READOUT y_t (1800ms) ---
        heroPhaseTimeouts.push(setTimeout(() => {
            if (travelerHead) travelerHead.style.left = `${railPositions[4]}%`;
            blocks.forEach((b, i) => b && b.classList.toggle("active-stage", i === 4));
            if (conns[3]) conns[3].classList.add("flow-active");

            // Probe readout using deterministic query vector q = [0.8, 0.4, 0.6]
            const qVec = [0.8, 0.4, 0.6];
            const num0 = (heroState[0][0] * qVec[0] + heroState[1][0] * qVec[1] + heroState[2][0] * qVec[2]);
            const num1 = (heroState[0][1] * qVec[0] + heroState[1][1] * qVec[1] + heroState[2][1] * qVec[2]);
            const denom = (heroNormalizer[0] * qVec[0] + heroNormalizer[1] * qVec[1] + heroNormalizer[2] * qVec[2]) || 1.0;

            const y0 = (num0 / denom).toFixed(2);
            const y1 = (num1 / denom).toFixed(2);
            const stateNorm = (heroState[0][0] + heroState[1][1] + heroState[2][2]).toFixed(2);

            const queryBox = document.getElementById("heroQueryReadout");
            if (queryBox) {
                queryBox.innerHTML = `<span class="text-accent-lavender font-mono">y_${heroTokenIndex + 1}=[${y0}, ${y1}]</span> (‖S‖=${stateNorm})`;
            }

            if (statusEl) {
                statusEl.textContent = `t = 0${heroTokenIndex + 1} | Query readout y_${heroTokenIndex + 1} extracted in O(1) time`;
            }

            // Increment token index
            heroTokenIndex++;
            if (heroTokenIndex >= HERO_TOKENS.length) {
                // Sequence completed
                heroPhaseTimeouts.push(setTimeout(() => {
                    finishHeroSequence();
                }, 400));
            }
        }, 1800));
    }

    window.heroEngine = {
        play: playHeroEngine,
        pause: pauseHeroEngine,
        replay: replayHeroEngine,
        getTokens: () => HERO_TOKENS,
        getIndex: () => heroTokenIndex,
        getState: () => heroState
    };


    // =========================================================================
    // 3. SECTION 02: PAIRWISE INTERACTION FOOTPRINT VISUALIZER
    // =========================================================================
    const sampleWords = ["The", "model", "reads", "one", "token", "at", "a", "time", "and", "updates", "memory", "state", "step", "by", "step", "now"];

    function initProblemVisualizer() {
        const presetButtons = document.querySelectorAll("#problem .btn-preset");
        presetButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                presetButtons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const n = parseInt(btn.getAttribute("data-n"), 10);
                renderPairwiseMatrix(n);
            });
        });

        // Initial render with N = 4
        renderPairwiseMatrix(4);
    }

    function renderPairwiseMatrix(N) {
        const tokenBar = document.getElementById("problemTokenBar");
        const colLabels = document.getElementById("matrixColLabels");
        const rowLabels = document.getElementById("matrixRowLabels");
        const grid = document.getElementById("pairwiseGrid");
        const canvas = document.getElementById("pairwiseDebtCanvas");
        const modeBadge = document.getElementById("debtModeBadge");
        const modeText = document.getElementById("debtModeText");

        const metricSeqN = document.getElementById("metricSeqN");
        const metricPairwiseTotal = document.getElementById("metricPairwiseTotal");
        const metricCausalSlots = document.getElementById("metricCausalSlots");
        const metricLinearUpdates = document.getElementById("metricLinearUpdates");
        const metricOverheadFactor = document.getElementById("metricOverheadFactor");

        // Metrics always calculate for exact N
        const totalSlots = N * N;
        const causalSlots = Math.round((N * (N + 1)) / 2);

        if (metricSeqN) metricSeqN.textContent = N;
        if (metricPairwiseTotal) metricPairwiseTotal.textContent = totalSlots.toLocaleString();
        if (metricCausalSlots) metricCausalSlots.textContent = causalSlots.toLocaleString();
        if (metricLinearUpdates) metricLinearUpdates.textContent = N.toLocaleString();
        if (metricOverheadFactor) metricOverheadFactor.textContent = `${N}.0×`;

        // Render token bar
        if (tokenBar) {
            tokenBar.innerHTML = "";
            if (N <= 16) {
                const activeWords = sampleWords.slice(0, N);
                activeWords.forEach((word, idx) => {
                    const sp = document.createElement("span");
                    sp.className = "token-token-box";
                    sp.textContent = `[${idx + 1}] ${word}`;
                    tokenBar.appendChild(sp);
                });
            } else {
                // Condensed display for large N
                const leadWords = sampleWords.slice(0, 6);
                leadWords.forEach((word, idx) => {
                    const sp = document.createElement("span");
                    sp.className = "token-token-box";
                    sp.textContent = `[${idx + 1}] ${word}`;
                    tokenBar.appendChild(sp);
                });
                const ellipsis = document.createElement("span");
                ellipsis.className = "token-token-box";
                ellipsis.textContent = `... (+${N - 8} more tokens) ...`;
                tokenBar.appendChild(ellipsis);
                const trailWords = sampleWords.slice(14, 16);
                trailWords.forEach((word, idx) => {
                    const sp = document.createElement("span");
                    sp.className = "token-token-box";
                    sp.textContent = `[${N - 1 + idx}] ${word}`;
                    tokenBar.appendChild(sp);
                });
            }
        }

        if (N <= 16) {
            // DIRECTLY RENDERED DOM GRID FOR SMALL N (<= 16)
            if (grid) grid.style.display = "grid";
            if (canvas) canvas.style.display = "none";
            if (modeBadge) modeBadge.style.display = "none";

            if (colLabels && rowLabels && grid) {
                colLabels.innerHTML = "";
                rowLabels.innerHTML = "";
                grid.innerHTML = "";

                grid.style.gridTemplateColumns = `repeat(${N}, 18px)`;

                const activeWords = sampleWords.slice(0, N);

                for (let j = 0; j < N; j++) {
                    const colLab = document.createElement("div");
                    colLab.className = "matrix-axis-label col";
                    colLab.textContent = `${j + 1}`;
                    colLab.title = `Key Token: ${activeWords[j]}`;
                    colLabels.appendChild(colLab);
                }

                for (let i = 0; i < N; i++) {
                    const rowLab = document.createElement("div");
                    rowLab.className = "matrix-axis-label";
                    rowLab.textContent = `${i + 1}`;
                    rowLab.title = `Query Token: ${activeWords[i]}`;
                    rowLabels.appendChild(rowLab);

                    for (let j = 0; j < N; j++) {
                        const cell = document.createElement("div");
                        cell.className = "matrix-cell";
                        cell.id = `cell-${i}-${j}`;
                        cell.title = `Comparison (${i+1}, ${j+1}): "${activeWords[i]}" ⟷ "${activeWords[j]}"`;

                        if (i === j) {
                            cell.style.background = "var(--accent-lavender)";
                        } else if (j > i) {
                            cell.classList.add("causal-masked");
                        } else {
                            const alpha = 0.25 + 0.5 * ((i - j) / N);
                            cell.style.background = `rgba(169, 154, 244, ${alpha})`;
                        }

                        grid.appendChild(cell);
                    }
                }
            }
        } else {
            // COMPRESSED CANVAS HEATMAP FOR LARGER N (> 16: e.g. 32, 64, 100)
            if (grid) grid.style.display = "none";
            if (canvas) canvas.style.display = "block";
            if (modeBadge) {
                modeBadge.style.display = "inline-flex";
                if (modeText) modeText.textContent = `Adaptive View: Compressed Heatmap Active (${N}×${N} = ${totalSlots.toLocaleString()} slots)`;
            }

            if (colLabels && rowLabels) {
                colLabels.innerHTML = `<span class="axis-range">Key Tokens: 1 . . . . . . . . . . . . ${N}</span>`;
                rowLabels.innerHTML = `<span class="axis-range">Query: 1<br>.<br>.<br>${N}</span>`;
            }

            drawPairwiseDebtCanvas(N);
        }
    }

    function drawPairwiseDebtCanvas(N) {
        const canvas = document.getElementById("pairwiseDebtCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#0B0B10";
        ctx.fillRect(0, 0, w, h);

        const cellW = w / N;
        const cellH = h / N;

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                if (i === j) {
                    ctx.fillStyle = "#C3B7FF";
                    ctx.fillRect(j * cellW, i * cellH, Math.max(1, cellW), Math.max(1, cellH));
                } else if (j > i) {
                    // Causal masked upper triangle
                    ctx.fillStyle = "rgba(17, 17, 26, 0.75)";
                    ctx.fillRect(j * cellW, i * cellH, Math.max(1, cellW), Math.max(1, cellH));
                } else {
                    // Causal computed lower triangle
                    const alpha = 0.2 + 0.6 * ((i - j) / N);
                    ctx.fillStyle = `rgba(169, 154, 244, ${alpha})`;
                    ctx.fillRect(j * cellW, i * cellH, Math.max(1, cellW), Math.max(1, cellH));
                }
            }
        }

        // Causal diagonal divider line
        ctx.strokeStyle = "rgba(195, 183, 255, 0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(w, h);
        ctx.stroke();
    }


    // =========================================================================
    // 4. SECTION 03: 100-TOKEN RACE (WATCH THE SEQUENCE GROW: 4 TO 100)
    // =========================================================================
    let currentRaceN = 16;
    let animatedPairwiseCount = 256;
    let animatedLinearCount = 16;
    let countAnimFrameId = null;

    function setSequenceLength(val) {
        const N = Math.max(4, Math.min(100, Math.round(Number(val))));
        currentRaceN = N;

        const slider = document.getElementById("pressureSlider");
        const sliderValDisplay = document.getElementById("sliderValDisplay");
        const pairwiseCountDisplay = document.getElementById("pairwiseCountDisplay");
        const linearCountDisplay = document.getElementById("linearCountDisplay");

        const tableCurrentN = document.getElementById("tableCurrentN");
        const tableStandardMem = document.getElementById("tableStandardMem");
        const tableLinearMem = document.getElementById("tableLinearMem");
        const canvasZoomTag = document.getElementById("canvasZoomTag");

        // Sync range slider input without dispatching redundant events
        if (slider && Number(slider.value) !== N) {
            slider.value = N;
        }

        // Immediately update all N-dependent displays (No interpolation of N!)
        if (sliderValDisplay) sliderValDisplay.textContent = `N = ${N}`;
        if (tableCurrentN) tableCurrentN.textContent = N;
        if (canvasZoomTag) canvasZoomTag.textContent = `Resolution: ${N}×${N} (${(N*N).toLocaleString()} slots)`;

        // Sync quick-jump buttons
        const jumps = document.querySelectorAll("#pressure .btn-jump");
        jumps.forEach(btn => {
            const v = parseInt(btn.getAttribute("data-val"), 10);
            btn.classList.toggle("active", v === N);
        });

        // Target counts for pairwise and linear
        const targetPairwise = N * N;
        const targetLinear = N;

        // Smoothly animate the count numbers only (towards target counts)
        if (countAnimFrameId) cancelAnimationFrame(countAnimFrameId);
        function animateCounts() {
            const stepP = (targetPairwise - animatedPairwiseCount) * 0.3;
            const stepL = (targetLinear - animatedLinearCount) * 0.3;

            if (Math.abs(stepP) > 1) {
                animatedPairwiseCount += Math.round(stepP);
            } else {
                animatedPairwiseCount = targetPairwise;
            }

            if (Math.abs(stepL) > 0.5) {
                animatedLinearCount += Math.round(stepL);
            } else {
                animatedLinearCount = targetLinear;
            }

            if (pairwiseCountDisplay) pairwiseCountDisplay.textContent = animatedPairwiseCount.toLocaleString();
            if (linearCountDisplay) linearCountDisplay.textContent = animatedLinearCount.toLocaleString();

            if (animatedPairwiseCount !== targetPairwise || animatedLinearCount !== targetLinear) {
                countAnimFrameId = requestAnimationFrame(animateCounts);
            }
        }
        animateCounts();

        // Memory Footprint (d = 64, FP16 = 2 bytes)
        const d = 64;
        const bytesPerFloat = 2;
        const standardCacheBytes = 2 * N * d * bytesPerFloat; // 256 * N bytes
        const linearStateBytes = d * d * bytesPerFloat;       // 8192 bytes = 8.0 KB constant

        const stdKB = (standardCacheBytes / 1024).toFixed(1);
        const linKB = (linearStateBytes / 1024).toFixed(1);

        if (tableStandardMem) tableStandardMem.textContent = `${stdKB} KB`;
        if (tableLinearMem) tableLinearMem.textContent = `${linKB} KB (Constant)`;

        // Redraw Canvas Heatmap & 1D Stream
        drawPairwiseCanvas(N);
        renderLinearStream(N);
    }
    window.setSequenceLength = setSequenceLength;
    window.getCurrentRaceN = () => currentRaceN;

    function initAttentionPressure() {
        const slider = document.getElementById("pressureSlider");
        const jumps = document.querySelectorAll("#pressure .btn-jump");

        if (slider) {
            slider.addEventListener("input", (e) => {
                setSequenceLength(e.target.value);
            });
        }

        jumps.forEach(btn => {
            btn.addEventListener("click", () => {
                const val = btn.getAttribute("data-val");
                setSequenceLength(val);
            });
        });

        // Initialize with default N = 16
        setSequenceLength(16);
    }

    function drawPairwiseCanvas(N) {
        const canvas = document.getElementById("pairwiseCanvas");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const width = canvas.width;
        const height = canvas.height;

        ctx.clearRect(0, 0, width, height);

        // Smooth cell sizing based on N (supporting up to N=100 effortlessly)
        const cellPixelSize = width / N;

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                if (i === j) {
                    ctx.fillStyle = "#C3B7FF";
                } else if (j > i) {
                    ctx.fillStyle = "#14131E";
                } else {
                    const heat = 0.25 + 0.65 * ((i - j) / N);
                    ctx.fillStyle = `rgba(248, 113, 113, ${heat})`;
                }
                ctx.fillRect(j * cellPixelSize, i * cellPixelSize, Math.max(1, cellPixelSize - 0.4), Math.max(1, cellPixelSize - 0.4));
            }
        }
    }

    function renderLinearStream(N) {
        const streamBox = document.getElementById("linearStreamBox");
        if (!streamBox) return;

        streamBox.innerHTML = "";
        const displayLimit = Math.min(N, 10);

        for (let t = 1; t <= displayLimit; t++) {
            const item = document.createElement("div");
            item.className = "stream-item";
            item.innerHTML = `
                <span class="stream-item-idx">t = ${String(t).padStart(2, '0')}</span>
                <span>Token x_${t}</span>
                <span class="stream-item-update">+ ΔS_${t}</span>
                <span class="text-accent-lavender font-mono">S_${t}</span>
            `;
            streamBox.appendChild(item);
        }

        if (N > displayLimit) {
            const summaryItem = document.createElement("div");
            summaryItem.className = "stream-item text-dim";
            summaryItem.style.justifyContent = "center";
            summaryItem.style.background = "rgba(169, 154, 244, 0.08)";
            summaryItem.style.border = "1px dashed rgba(169, 154, 244, 0.3)";
            summaryItem.innerHTML = `<span>+ ${N - displayLimit} more sequential updates \(\to\) Total: <strong>${N} state updates</strong> to fixed \(S\)</span>`;
            streamBox.appendChild(summaryItem);
        }
    }


    // =========================================================================
    // 5. SECTION 04: TOY LINEAR ATTENTION ENGINE (Visually Central Substrate)
    // =========================================================================
    const ENGINE_STATE = {
        scenarioKey: "associative",
        stepIndex: 0,
        stateMatrix: createZeroMatrix(4, 4),
        normalizerVector: [0, 0, 0, 0],
        tokenHistory: [],
        isRunning: true, // Default: AUTO RUN active!
        timerId: null,
        speedMultiplier: 1.0,
        detailsOpen: false
    };
    window.ENGINE_STATE = ENGINE_STATE;

    function initToyExperiment() {
        const patternSelect = document.getElementById("patternSelect");
        const btnStepNext = document.getElementById("btnStepNext");
        const btnAutoRun = document.getElementById("btnAutoRun");
        const btnResetExp = document.getElementById("btnResetExp");
        const btnToggleDetails = document.getElementById("btnToggleDetails");
        const expSpeedSlider = document.getElementById("expSpeedSlider");
        const speedVal = document.getElementById("speedVal");
        const btnClearLog = document.getElementById("btnClearLog");
        const btnProbeQuery = document.getElementById("btnProbeQuery");

        if (patternSelect) {
            patternSelect.addEventListener("change", (e) => {
                ENGINE_STATE.scenarioKey = e.target.value;
                resetExperiment();
            });
        }

        if (btnAutoRun) {
            btnAutoRun.addEventListener("click", () => {
                if (ENGINE_STATE.isRunning) {
                    pauseAutoRun();
                } else {
                    startAutoRun();
                }
            });
        }

        if (btnStepNext) {
            btnStepNext.addEventListener("click", () => {
                pauseAutoRun();
                stepExperiment();
            });
        }

        if (btnResetExp) {
            btnResetExp.addEventListener("click", () => {
                resetExperiment();
            });
        }

        if (btnToggleDetails) {
            btnToggleDetails.addEventListener("click", () => {
                ENGINE_STATE.detailsOpen = !ENGINE_STATE.detailsOpen;
                const drawer = document.getElementById("detailedMathDrawer");
                const label = document.getElementById("toggleDetailsLabel");
                if (drawer) drawer.style.display = ENGINE_STATE.detailsOpen ? "block" : "none";
                if (label) label.textContent = ENGINE_STATE.detailsOpen ? "HIDE MATH ▲" : "INSPECT MATH ▼";
            });
        }

        if (expSpeedSlider && speedVal) {
            expSpeedSlider.addEventListener("input", (e) => {
                const s = parseFloat(e.target.value);
                ENGINE_STATE.speedMultiplier = s;
                speedVal.textContent = `${s.toFixed(1)}×`;
                if (ENGINE_STATE.isRunning) {
                    pauseAutoRun();
                    startAutoRun();
                }
            });
        }

        if (btnClearLog) {
            btnClearLog.addEventListener("click", () => {
                const logWin = document.getElementById("eventLogWindow");
                if (logWin) logWin.innerHTML = "";
            });
        }

        if (btnProbeQuery) {
            btnProbeQuery.addEventListener("click", () => {
                evaluateQueryProbe();
            });
        }

        // Spacebar hotkey for Step
        window.addEventListener("keydown", (e) => {
            if (e.code === "Space" && e.target === document.body) {
                e.preventDefault();
                pauseAutoRun();
                stepExperiment();
            }
        });

        // Initialize state & start Auto Run
        resetExperiment();
        startAutoRun();
    }

    function resetExperiment() {
        pauseAutoRun();
        ENGINE_STATE.stepIndex = 0;
        ENGINE_STATE.stateMatrix = createZeroMatrix(4, 4);
        ENGINE_STATE.normalizerVector = [0, 0, 0, 0];
        ENGINE_STATE.tokenHistory = [];

        renderTimelineRibbon();
        renderCoreFlow(null);
        renderDetailedVectors(null);
        renderDeltaMatrix(createZeroMatrix(4, 4));
        renderStateMatrix(ENGINE_STATE.stateMatrix);

        const logWin = document.getElementById("eventLogWindow");
        if (logWin) {
            logWin.innerHTML = `<div class="log-entry text-dim"><span>[SYS]</span> Recurrent state S initialized to 0. Ready for step 1.</div>`;
        }

        evaluateQueryProbe();
    }

    function stepExperiment() {
        const scenario = EXPERIMENT_SCENARIOS[ENGINE_STATE.scenarioKey];
        const tokens = scenario.tokens;

        if (ENGINE_STATE.stepIndex >= tokens.length) {
            ENGINE_STATE.stepIndex = 0;
        }

        const currentToken = tokens[ENGINE_STATE.stepIndex];
        const t = ENGINE_STATE.stepIndex + 1;

        // 1. Feature Map \phi(k_t)
        const phiK = featureMapELU(currentToken.k);

        // 2. Outer Product \Delta S_t = \phi(k_t) \otimes v_t^T
        const deltaS = outerProduct(phiK, currentToken.v);
        const deltaNorm = frobeniusNorm(deltaS);

        // 3. Accumulate State Matrix S_t = S_{t-1} + \Delta S_t
        ENGINE_STATE.stateMatrix = addMatrices(ENGINE_STATE.stateMatrix, deltaS);
        const stateNorm = frobeniusNorm(ENGINE_STATE.stateMatrix);

        // 4. Accumulate Normalizer z_t = z_{t-1} + \phi(k_t)
        for (let i = 0; i < 4; i++) {
            ENGINE_STATE.normalizerVector[i] += phiK[i];
        }

        // Save into history
        ENGINE_STATE.tokenHistory.push({
            t: t,
            token: currentToken,
            phiK: phiK,
            deltaS: deltaS,
            deltaNorm: deltaNorm,
            stateSnapshot: JSON.parse(JSON.stringify(ENGINE_STATE.stateMatrix)),
            stateNorm: stateNorm
        });

        // Trigger visual pulse transitions across the flow
        triggerFlowPulses();

        // Update UI components
        renderTimelineRibbon();
        renderCoreFlow(currentToken, t, deltaNorm, stateNorm);
        renderDetailedVectors(currentToken, phiK);
        renderDeltaMatrix(deltaS, deltaNorm);
        renderStateMatrix(ENGINE_STATE.stateMatrix, stateNorm);
        appendTelemetryLog(t, currentToken.word, deltaNorm, stateNorm);
        evaluateQueryProbe();

        ENGINE_STATE.stepIndex++;
    }

    function triggerFlowPulses() {
        const colToken = document.getElementById("flowColToken");
        const colUpdate = document.getElementById("flowColUpdate");
        const colState = document.getElementById("flowColState");
        const arrow1 = document.getElementById("flowArrow1");
        const arrow2 = document.getElementById("flowArrow2");

        if (colToken) {
            colToken.classList.add("flash-step");
            setTimeout(() => colToken.classList.remove("flash-step"), 300);
        }

        setTimeout(() => {
            if (arrow1) arrow1.classList.add("pulse-glow");
            if (colUpdate) colUpdate.classList.add("flash-step");
            setTimeout(() => {
                if (arrow1) arrow1.classList.remove("pulse-glow");
                if (colUpdate) colUpdate.classList.remove("flash-step");
            }, 300);
        }, 150);

        setTimeout(() => {
            if (arrow2) arrow2.classList.add("pulse-glow");
            if (colState) colState.classList.add("flash-step");
            setTimeout(() => {
                if (arrow2) arrow2.classList.remove("pulse-glow");
                if (colState) colState.classList.remove("flash-step");
            }, 300);
        }, 300);
    }

    function startAutoRun() {
        ENGINE_STATE.isRunning = true;
        const icon = document.getElementById("autoRunIcon");
        const label = document.getElementById("autoRunLabel");

        if (label) label.textContent = "PAUSE";
        if (icon) {
            icon.innerHTML = `<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>`;
        }

        const interval = Math.max(500, 1600 / ENGINE_STATE.speedMultiplier);
        ENGINE_STATE.timerId = setInterval(() => {
            stepExperiment();
        }, interval);
    }

    function pauseAutoRun() {
        ENGINE_STATE.isRunning = false;
        if (ENGINE_STATE.timerId) {
            clearInterval(ENGINE_STATE.timerId);
            ENGINE_STATE.timerId = null;
        }
        const icon = document.getElementById("autoRunIcon");
        const label = document.getElementById("autoRunLabel");
        if (label) label.textContent = "AUTO RUN";
        if (icon) {
            icon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"/>`;
        }
    }

    function renderTimelineRibbon() {
        const strip = document.getElementById("tokensFlowStrip");
        if (!strip) return;
        strip.innerHTML = "";

        const scenario = EXPERIMENT_SCENARIOS[ENGINE_STATE.scenarioKey];
        scenario.tokens.forEach((tok, idx) => {
            const node = document.createElement("div");
            const isProcessed = idx < ENGINE_STATE.stepIndex;
            const isCurrent = idx === (ENGINE_STATE.stepIndex - 1);

            node.className = `timeline-token-node ${isProcessed ? 'processed' : ''} ${isCurrent ? 'current' : ''}`;
            node.innerHTML = `<span>[${idx + 1}]</span> <strong>${tok.word}</strong>`;
            node.title = `${tok.word} (${tok.category})`;

            node.addEventListener("click", () => {
                pauseAutoRun();
                ENGINE_STATE.stepIndex = idx;
                stepExperiment();
            });

            strip.appendChild(node);
        });
    }

    function renderCoreFlow(tok, t = 1, deltaNorm = 0, stateNorm = 0) {
        const coreWord = document.getElementById("coreTokenWord");
        const coreCat = document.getElementById("coreTokenCat");
        const coreIdx = document.getElementById("coreTokenIdx");
        const deltaNormVal = document.getElementById("deltaNormVal");
        const stateNormVal = document.getElementById("stateNormVal");

        if (!tok) {
            if (coreWord) coreWord.textContent = '"—"';
            if (coreCat) coreCat.textContent = 'Waiting for input';
            if (coreIdx) coreIdx.textContent = 'Step t = 00';
            if (deltaNormVal) deltaNormVal.textContent = '0.000';
            if (stateNormVal) stateNormVal.textContent = '0.000';
            return;
        }

        if (coreWord) coreWord.textContent = `"${tok.word}"`;
        if (coreCat) coreCat.textContent = tok.category;
        if (coreIdx) coreIdx.textContent = `Step t = ${String(t).padStart(2, '0')}`;
        if (deltaNormVal) deltaNormVal.textContent = deltaNorm.toFixed(3);
        if (stateNormVal) stateNormVal.textContent = stateNorm.toFixed(3);
    }

    function renderDetailedVectors(tok, phiK = null) {
        const vecK = document.getElementById("vecKDisplay");
        const vecPhi = document.getElementById("vecPhiDisplay");
        const vecV = document.getElementById("vecVDisplay");

        if (!tok) {
            if (vecK) vecK.textContent = '[0.00, 0.00, 0.00, 0.00]';
            if (vecPhi) vecPhi.textContent = '[0.00, 0.00, 0.00, 0.00]';
            if (vecV) vecV.textContent = '[0.00, 0.00, 0.00, 0.00]';
            return;
        }

        if (vecK) vecK.textContent = `[${tok.k.map(v => v.toFixed(2)).join(", ")}]`;
        if (vecPhi) vecPhi.textContent = `[${phiK.map(v => v.toFixed(2)).join(", ")}]`;
        if (vecV) vecV.textContent = `[${tok.v.map(v => v.toFixed(2)).join(", ")}]`;
    }

    function renderDeltaMatrix(deltaS, norm = 0) {
        const box = document.getElementById("deltaMatrixBox");
        if (!box) return;

        box.innerHTML = "";
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                const val = deltaS[i][j];
                const cell = document.createElement("div");
                cell.className = "num-cell updated-flash";
                cell.textContent = val.toFixed(1);
                const alpha = Math.min(1, Math.max(0.12, val * 0.8));
                cell.style.background = `rgba(195, 183, 255, ${alpha})`;
                box.appendChild(cell);
            }
        }
    }

    function renderStateMatrix(stateMat, norm = 0) {
        const box = document.getElementById("stateMatrixBox");
        if (!box) return;

        box.innerHTML = "";
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                const val = stateMat[i][j];
                const cell = document.createElement("div");
                cell.className = "num-cell";
                cell.textContent = val.toFixed(1);
                const alpha = Math.min(1, Math.max(0.15, (val / 3.0)));
                cell.style.background = `rgba(169, 154, 244, ${alpha})`;
                cell.title = `S[${i}, ${j}] = ${val.toFixed(3)}`;
                box.appendChild(cell);
            }
        }
    }

    function appendTelemetryLog(t, word, deltaNorm, stateNorm) {
        const logWin = document.getElementById("eventLogWindow");
        if (!logWin) return;

        const entry = document.createElement("div");
        entry.className = "log-entry";
        entry.innerHTML = `
            <span class="log-time">[t=${String(t).padStart(2, '0')}]</span>
            <span class="log-token">token="${word}"</span>
            <span class="log-delta">memory updated (||ΔS||=${deltaNorm.toFixed(2)})</span>
            <span class="text-accent-lavender">total ||S||=${stateNorm.toFixed(2)}</span>
        `;
        logWin.appendChild(entry);
        logWin.scrollTop = logWin.scrollHeight;
    }

    function evaluateQueryProbe() {
        const select = document.getElementById("queryPresetSelect");
        const barsBox = document.getElementById("retrievedBarsContainer");
        const interpBox = document.getElementById("probeInterpretation");

        if (!select || !barsBox) return;

        const probeType = select.value;
        let queryK = [0.85, -0.10, 0.40, 0.15]; // apple

        if (probeType === "second") {
            queryK = [-0.20, 0.15, 0.85, 0.20]; // wolf
        } else if (probeType === "neutral") {
            queryK = [0.25, 0.25, 0.25, 0.25];
        }

        const phiQ = featureMapELU(queryK);
        const numVec = vecMatProduct(phiQ, ENGINE_STATE.stateMatrix);
        const denom = dotProduct(phiQ, ENGINE_STATE.normalizerVector);

        const outVec = (denom > 1e-6) ? numVec.map(v => v / denom) : [0, 0, 0, 0];

        barsBox.innerHTML = "";
        const dimLabels = ["Dim 1 (Fruit)", "Dim 2 (Rel)", "Dim 3 (Animal)", "Dim 4 (Gem)"];
        for (let i = 0; i < 4; i++) {
            const val = Math.max(0, outVec[i]);
            const pct = Math.min(100, Math.round(val * 100));
            const row = document.createElement("div");
            row.className = "retrieved-bar-row";
            row.innerHTML = `
                <span class="bar-dim-name">${dimLabels[i]}</span>
                <div class="bar-track"><div class="bar-fill" style="width: ${pct}%"></div></div>
                <span class="bar-val font-mono">${val.toFixed(2)}</span>
            `;
            barsBox.appendChild(row);
        }

        if (interpBox) {
            if (denom < 1e-6) {
                interpBox.innerHTML = `State is empty. Process tokens to observe content-addressable readout.`;
            } else {
                interpBox.innerHTML = `
                    Query probing: Top associative retrieval activation is on <strong>${outVec[0] > outVec[2] ? 'Dim 1 (Fruit)' : 'Dim 3 (Animal)'}</strong> (${Math.max(...outVec).toFixed(2)}). Evaluated via \(\frac{\phi(q)^T S_t}{\phi(q)^T z_t}\).
                `;
            }
        }
    }


    // =========================================================================
    // 6. SECTION 05: PREDICTION CHALLENGE (UPDATED FOR 4-100 RANGE)
    // =========================================================================
    function initPredictionChallenge() {
        // Question 1
        const block1 = document.getElementById("quizBlock1");
        if (block1) {
            const opts1 = block1.querySelectorAll(".quiz-opt");
            opts1.forEach(btn => {
                btn.addEventListener("click", () => {
                    const isCorrect = btn.getAttribute("data-correct") === "true";
                    opts1.forEach(b => {
                        b.disabled = true;
                        b.classList.remove("user-picked");
                        if (b.getAttribute("data-correct") === "true") {
                            b.classList.add("opt-correct");
                        } else if (b === btn) {
                            b.classList.add("opt-incorrect");
                        }
                    });
                    btn.classList.add("user-picked");

                    const panel = document.getElementById("truthPanel1");
                    const expectedVal = document.getElementById("expectedVal1");
                    const expectedDesc = document.getElementById("expectedDesc1");

                    if (panel && expectedVal && expectedDesc) {
                        panel.style.display = "block";
                        expectedVal.textContent = btn.querySelector(".opt-text").textContent;
                        expectedDesc.textContent = isCorrect 
                            ? "Correct! You anticipated the quadratic scaling penalty of traditional attention." 
                            : "Notice the quadratic gap: 30 tokens evaluated pairwise requires 30 × 30 = 900 slots, compared to 30 sequential updates.";
                    }
                });
            });
        }

        // Question 2 (N = 50 -> 100)
        const block2 = document.getElementById("quizBlock2");
        if (block2) {
            const opts2 = block2.querySelectorAll(".quiz-opt");
            opts2.forEach(btn => {
                btn.addEventListener("click", () => {
                    const isCorrect = btn.getAttribute("data-correct") === "true";
                    opts2.forEach(b => {
                        b.disabled = true;
                        b.classList.remove("user-picked");
                        if (b.getAttribute("data-correct") === "true") {
                            b.classList.add("opt-correct");
                        } else if (b === btn) {
                            b.classList.add("opt-incorrect");
                        }
                    });
                    btn.classList.add("user-picked");

                    const panel = document.getElementById("truthPanel2");
                    const expectedVal = document.getElementById("expectedVal2");
                    const expectedDesc = document.getElementById("expectedDesc2");

                    if (panel && expectedVal && expectedDesc) {
                        panel.style.display = "block";
                        expectedVal.textContent = btn.querySelector(".opt-text").textContent;
                        expectedDesc.textContent = isCorrect 
                            ? "Correct! Doubling N quadruples pairwise comparison slots (50² = 2,500 to 100² = 10,000)." 
                            : "Remember quadratic scaling: (2N)² = 4N². Pairwise slots quadruple from 2,500 to 10,000 slots!";
                    }
                });
            });
        }
    }


    // =========================================================================
    // 7. SECTION 06: BDH / BDH-CQ TABS
    // =========================================================================
    function initBdhTabs() {
        const tabBtns = document.querySelectorAll(".tab-btn");
        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");

                tabBtns.forEach(b => {
                    b.classList.remove("active");
                    b.setAttribute("aria-selected", "false");
                });
                btn.classList.add("active");
                btn.setAttribute("aria-selected", "true");

                document.querySelectorAll(".tab-pane").forEach(pane => {
                    pane.classList.remove("active");
                });
                const targetPane = document.getElementById(`tabContent${targetTab}`);
                if (targetPane) targetPane.classList.add("active");
            });
        });
    }


    // =========================================================================
    // 8. SCROLL SECTION TRACKING & PROGRESS RAIL
    // =========================================================================
    function initScrollTracking() {
        const sections = document.querySelectorAll(".section-panel");
        const railDots = document.querySelectorAll(".rail-dot");
        const navItems = document.querySelectorAll(".nav-item");

        function onScroll() {
            const scrollPos = window.scrollY + window.innerHeight * 0.35;

            sections.forEach((sec, idx) => {
                const top = sec.offsetTop;
                const height = sec.offsetHeight;

                if (scrollPos >= top && scrollPos < top + height) {
                    railDots.forEach(dot => dot.classList.remove("active"));
                    if (railDots[idx]) railDots[idx].classList.add("active");

                    navItems.forEach(item => item.classList.remove("active"));
                    if (navItems[idx]) navItems[idx].classList.add("active");
                }
            });
        }

        window.addEventListener("scroll", onScroll, { passive: true });
        onScroll();
    }


    // =========================================================================
    // 9. SECTION 08: 60-SECOND EXPLAINER CHALLENGE & VERIFICATION
    // =========================================================================
    let explainerSecondsLeft = 60;
    let explainerTimerId = null;

    function init60sExplainer() {
        const timerBtn = document.getElementById("explainerTimerBtn");
        const timerBtnText = document.getElementById("explainerTimerBtnText");
        const timerDisplay = document.getElementById("explainerTimerDisplay");
        const verifyBtn = document.getElementById("verifyPipelineBtn");
        const verdict = document.getElementById("pipelineVerdict");
        const cards = document.querySelectorAll(".pipeline-challenge-card");

        if (timerBtn) {
            timerBtn.addEventListener("click", () => {
                if (explainerTimerId) {
                    // Pause
                    clearInterval(explainerTimerId);
                    explainerTimerId = null;
                    if (timerBtnText) timerBtnText.textContent = "RESUME TIMER";
                } else {
                    if (explainerSecondsLeft <= 0) explainerSecondsLeft = 60;
                    if (timerBtnText) timerBtnText.textContent = "PAUSE TIMER";

                    explainerTimerId = setInterval(() => {
                        explainerSecondsLeft--;
                        const mins = Math.floor(explainerSecondsLeft / 60);
                        const secs = explainerSecondsLeft % 60;
                        if (timerDisplay) {
                            timerDisplay.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
                        }

                        if (explainerSecondsLeft <= 0) {
                            clearInterval(explainerTimerId);
                            explainerTimerId = null;
                            if (timerBtnText) timerBtnText.textContent = "RESET 60s";
                            if (timerDisplay) timerDisplay.textContent = "00:00";
                        }
                    }, 1000);
                }
            });
        }

        if (verifyBtn) {
            verifyBtn.addEventListener("click", () => {
                // Progressively illuminate the 4 cards in causal order: Tokens -> Update -> State -> Query
                cards.forEach(c => c.classList.remove("card-illuminated"));
                if (verdict) verdict.style.display = "none";

                cards.forEach((card, idx) => {
                    setTimeout(() => {
                        card.classList.add("card-illuminated");
                        if (idx === cards.length - 1 && verdict) {
                            verdict.style.display = "block";
                        }
                    }, idx * 300);
                });
            });
        }
    }


    // =========================================================================
    // 10. REDUCED MOTION TOGGLE & ACCESSIBILITY
    // =========================================================================
    function initAccessibility() {
        const toggleBtn = document.getElementById("toggleMotionBtn");
        if (toggleBtn) {
            toggleBtn.addEventListener("click", () => {
                document.body.classList.toggle("reduced-motion");
                const isReduced = document.body.classList.contains("reduced-motion");
                toggleBtn.setAttribute("aria-pressed", isReduced ? "true" : "false");
                toggleBtn.title = isReduced ? "Enable Animations" : "Reduce Motion";
            });
        }
    }


    // =========================================================================
    // DOM READY INITIALIZATION
    // =========================================================================
    document.addEventListener("DOMContentLoaded", () => {
        initHeroEngine();
        initProblemVisualizer();
        initAttentionPressure();
        initToyExperiment();
        initPredictionChallenge();
        initBdhTabs();
        init60sExplainer();
        initScrollTracking();
        initAccessibility();
    });

})();