"""Synthetic corpus and query generation for the subject 9 demo.

The corpus is generated rather than downloaded so experiments are reproducible
and the relevance labels are known. This is important for precision@K and
recall@K: without labels, we would only be showing anecdotal search results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random


RANDOM_STATE = 42


SEMANTIC_GROUPS: dict[str, list[str]] = {
    "semantic_search": [
        "semantic retrieval",
        "meaning based lookup",
        "conceptual lookup",
        "conceptual retrieval",
        "conceptual inference",
        "dense retrieval",
        "vector search",
        "neural document search",
        "embedding search",
    ],
    "bm25": [
        "bm25 ranking",
        "lexical search",
        "keyword matching",
        "term frequency scoring",
    ],
    "patient_triage": [
        "patient triage",
        "clinical prioritization",
        "emergency assessment",
        "hospital routing",
    ],
    "medical_imaging": [
        "medical imaging",
        "radiology screening",
        "xray analysis",
        "scan interpretation",
    ],
    "energy_forecasting": [
        "energy forecasting",
        "solar production prediction",
        "wind power forecasting",
        "renewable output prediction",
    ],
    "time_series": [
        "time series",
        "temporal signal",
        "seasonal pattern",
        "sequential measurement",
    ],
    "anomaly_detection": [
        "anomaly detection",
        "outlier detection",
        "rare event monitoring",
        "irregular pattern discovery",
    ],
    "cyber_threat": [
        "cyber threat detection",
        "intrusion monitoring",
        "malware campaign analysis",
        "network attack discovery",
    ],
    "fraud_detection": [
        "fraud detection",
        "suspicious transaction screening",
        "payment abuse discovery",
        "transaction risk monitoring",
    ],
    "risk_scoring": [
        "risk scoring",
        "credit risk assessment",
        "default probability estimation",
        "portfolio risk ranking",
    ],
    "graph_recommendation": [
        "graph recommendation",
        "collaborative filtering",
        "user item ranking",
        "personalized recommendation",
    ],
    "user_clustering": [
        "user clustering",
        "audience segmentation",
        "behavioral grouping",
        "customer cohort discovery",
    ],
    "robot_navigation": [
        "robot navigation",
        "autonomous path planning",
        "obstacle avoidance",
        "mobile robot routing",
    ],
    "sensor_fusion": [
        "sensor fusion",
        "multi sensor perception",
        "camera lidar fusion",
        "signal integration",
    ],
    "dna_alignment": [
        "dna alignment",
        "genomic sequence matching",
        "variant discovery",
        "gene sequence comparison",
    ],
    "document_classification": [
        "document classification",
        "case file tagging",
        "text categorization",
        "legal document labeling",
    ],
    "citation_retrieval": [
        "citation retrieval",
        "case law lookup",
        "precedent search",
        "legal reference discovery",
    ],
    "ranking": [
        "ranking",
        "result ordering",
        "top k selection",
        "relevance sorting",
    ],
}

QUERY_ONLY_PHRASES = {
    "conceptual lookup",
    "conceptual retrieval",
    "conceptual inference",
}


TOPICS: dict[str, dict[str, object]] = {
    "healthcare_ai": {
        "label": "Healthcare AI",
        "domain_terms": ["hospital", "clinical", "patient", "diagnosis", "care pathway"],
        "concepts": [
            "patient_triage",
            "medical_imaging",
            "anomaly_detection",
            "semantic_search",
        ],
    },
    "climate_energy": {
        "label": "Climate and energy",
        "domain_terms": ["grid", "renewable", "weather", "electricity", "forecast"],
        "concepts": [
            "energy_forecasting",
            "time_series",
            "anomaly_detection",
            "semantic_search",
        ],
    },
    "cybersecurity": {
        "label": "Cybersecurity",
        "domain_terms": ["network", "endpoint", "traffic", "alert", "incident"],
        "concepts": [
            "cyber_threat",
            "anomaly_detection",
            "semantic_search",
            "ranking",
        ],
    },
    "finance_risk": {
        "label": "Finance and risk",
        "domain_terms": ["bank", "payment", "portfolio", "credit", "market"],
        "concepts": [
            "fraud_detection",
            "risk_scoring",
            "time_series",
            "anomaly_detection",
        ],
    },
    "recommender_systems": {
        "label": "Recommender systems",
        "domain_terms": ["catalog", "profile", "session", "preference", "content"],
        "concepts": [
            "graph_recommendation",
            "user_clustering",
            "semantic_search",
            "bm25",
            "ranking",
        ],
    },
    "robotics": {
        "label": "Robotics",
        "domain_terms": ["robot", "warehouse", "trajectory", "camera", "lidar"],
        "concepts": [
            "robot_navigation",
            "sensor_fusion",
            "time_series",
            "anomaly_detection",
        ],
    },
    "bioinformatics": {
        "label": "Bioinformatics",
        "domain_terms": ["genome", "protein", "sequence", "mutation", "cohort"],
        "concepts": [
            "dna_alignment",
            "semantic_search",
            "user_clustering",
            "anomaly_detection",
        ],
    },
    "legal_tech": {
        "label": "Legal technology",
        "domain_terms": ["contract", "court", "case", "clause", "law"],
        "concepts": [
            "semantic_search",
            "bm25",
            "document_classification",
            "citation_retrieval",
            "ranking",
        ],
    },
}


METHODS = [
    "contrastive learning",
    "probabilistic modeling",
    "graph based indexing",
    "feature hashing",
    "nearest neighbor search",
    "Bayesian calibration",
    "sequence modeling",
    "representation learning",
]

OUTCOMES = [
    "lower latency",
    "better recall",
    "more stable decisions",
    "interpretable ranking",
    "robust retrieval",
    "faster filtering",
]

LIMITS = [
    "sensitivity to noisy labels",
    "memory growth",
    "domain drift",
    "ambiguous queries",
    "rare terminology",
    "cold start documents",
]

TITLE_TEMPLATES = [
    "{concept_a} for {domain} data",
    "{method} improves {concept_a}",
    "A benchmark of {concept_a} and {concept_b}",
    "{domain} applications of {concept_a}",
]

ABSTRACT_TEMPLATES = [
    (
        "This study investigates {concept_a} in {domain} datasets. "
        "The approach combines {method} with {concept_b} to obtain {outcome}. "
        "Experiments highlight {limit} as the main limitation."
    ),
    (
        "We present a pipeline for {concept_a} over {domain} records. "
        "The system uses {method}, compares several ranking strategies, and reports {outcome}. "
        "The analysis also discusses {limit}."
    ),
    (
        "The paper studies {concept_a} when {domain} collections become large. "
        "It introduces synthetic benchmarks, evaluates {concept_b}, and measures {outcome}. "
        "Results remain dependent on {limit}."
    ),
]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    topic: str
    topic_label: str
    concepts: tuple[str, ...]
    abstract: str
    domain: str = ""

    @property
    def primary_concept(self) -> str:
        """Concept that drives the title (``concept_a``).

        The corpus generator always renders ``concepts[0]`` in the title, so it is
        the single most salient concept of the document. Graded relevance treats a
        match on this concept as stronger than a match on the secondary concept.
        """

        return self.concepts[0] if self.concepts else ""

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}"

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["concepts"] = list(self.concepts)
        row["text"] = self.text
        return row


# A document is "binary relevant" (for precision@K / recall@K / MAP) when its
# graded relevance reaches this threshold. Grade 3 means the document matches the
# query on all three facets (topic, primary concept, domain), which keeps the gold
# set small and discriminating (~5-30 docs / query on 2000) instead of saturating.
BINARY_RELEVANCE_THRESHOLD = 3


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_concepts: tuple[str, ...]
    topic: str | None = None
    note: str = ""
    relevant_domain: str | None = None
    style: str = "mixed"

    def relevance_grade(self, document: Document) -> int:
        """Graded relevance in ``{0, 1, 2, 3}`` for a (query, document) pair.

        The grade counts how many *facets* of the information need the document
        satisfies, from coarse to fine:

        - **topic** : same application area (``document.topic == self.topic``);
        - **concept** : the query concept appears in ``document.concepts``;
        - **primary concept** : that concept is the *salient* one (the title
          concept ``document.concepts[0]``);
        - **domain** : the document uses the exact domain term the query targets.

        Grades:

        - ``3`` : same topic **and** the query concept is the document's primary
          concept **and** the domain term matches — the unambiguous gold answer;
        - ``2`` : same topic and primary concept, but a different domain term;
        - ``1`` : thematically adjacent — the concept is present as a *secondary*
          concept of the right topic, or it is the primary concept of a *different*
          topic (concepts such as ``anomaly_detection`` are shared across topics);
        - ``0`` : not relevant.

        Only grade ``3`` counts as binary-relevant (see
        :data:`BINARY_RELEVANCE_THRESHOLD`); grades ``1``/``2`` provide the partial
        credit that makes nDCG@10 discriminate between engines.
        """

        if not self.relevant_concepts:
            return 0
        concept = self.relevant_concepts[0]
        document_concepts = set(document.concepts)
        concept_present = concept in document_concepts
        primary_match = document.primary_concept == concept
        topic_match = self.topic is None or document.topic == self.topic
        domain_match = self.relevant_domain is None or document.domain == self.relevant_domain

        if topic_match and primary_match and domain_match:
            return 3
        if topic_match and primary_match:
            return 2
        if (topic_match and concept_present) or (primary_match and not topic_match):
            return 1
        return 0

    def is_relevant(self, document: Document) -> bool:
        """Binary relevance used by precision@K / recall@K / MAP."""

        return self.relevance_grade(document) >= BINARY_RELEVANCE_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["relevant_concepts"] = list(self.relevant_concepts)
        return row


def _choice_phrase(rng: random.Random, concept: str) -> str:
    phrases = [phrase for phrase in SEMANTIC_GROUPS[concept] if phrase not in QUERY_ONLY_PHRASES]
    return rng.choice(phrases)


def generate_corpus(n_docs: int = 2_000, random_state: int = RANDOM_STATE) -> list[Document]:
    """Generate a labeled synthetic corpus of research-style abstracts."""

    rng = random.Random(random_state)
    topic_names = list(TOPICS)
    documents: list[Document] = []

    for idx in range(n_docs):
        topic = topic_names[idx % len(topic_names)]
        spec = TOPICS[topic]
        concepts = tuple(rng.sample(spec["concepts"], k=2))  # type: ignore[arg-type]
        concept_a = _choice_phrase(rng, concepts[0])
        concept_b = _choice_phrase(rng, concepts[1])
        domain = rng.choice(spec["domain_terms"])  # type: ignore[arg-type]
        method = rng.choice(METHODS)
        outcome = rng.choice(OUTCOMES)
        limit = rng.choice(LIMITS)
        title = rng.choice(TITLE_TEMPLATES).format(
            concept_a=concept_a.title(),
            concept_b=concept_b.title(),
            domain=str(domain).title(),
            method=method.title(),
        )
        abstract = rng.choice(ABSTRACT_TEMPLATES).format(
            concept_a=concept_a,
            concept_b=concept_b,
            domain=domain,
            method=method,
            outcome=outcome,
            limit=limit,
        )
        documents.append(
            Document(
                doc_id=f"D{idx:05d}",
                title=title,
                topic=topic,
                topic_label=str(spec["label"]),
                concepts=concepts,
                abstract=abstract,
                domain=str(domain),
            )
        )

    rng.shuffle(documents)
    return documents


def default_queries() -> list[Query]:
    """Evaluation queries spanning the lexical-vs-semantic spectrum.

    Each query targets one ``(topic, primary concept, domain term)`` triple, so its
    gold set (grade 3) stays small and discriminating. The ``style`` field tags the
    expected winner so the hybrid analysis can be read per regime:

    - ``lexical`` : the query reuses the exact concept vocabulary -> BM25-friendly;
    - ``semantic`` : the concept is paraphrased (different words, same meaning) ->
      embedding-friendly, while the domain term still lets BM25 pre-filter;
    - ``mixed`` : a blend of exact and paraphrased terms.
    """

    return [
        Query(
            "Q01",
            "retrieve case documents that express the same idea in other words",
            ("semantic_search",),
            "legal_tech",
            note="Paraphrase of semantic retrieval; should favor embeddings over keywords.",
            relevant_domain="case",
            style="semantic",
        ),
        Query(
            "Q02",
            "precedent search across court decisions and case law",
            ("citation_retrieval",),
            "legal_tech",
            note="Domain-specific legal lookup with partly exact vocabulary.",
            relevant_domain="court",
            style="mixed",
        ),
        Query(
            "Q03",
            "bm25 keyword matching and term frequency scoring over the catalog",
            ("bm25",),
            "recommender_systems",
            note="Lexical control query: exact BM25 vocabulary.",
            relevant_domain="catalog",
            style="lexical",
        ),
        Query(
            "Q04",
            "hospital emergency assessment and patient triage routing",
            ("patient_triage",),
            "healthcare_ai",
            relevant_domain="hospital",
            style="lexical",
        ),
        Query(
            "Q05",
            "reading and interpreting radiology scans for diagnosis",
            ("medical_imaging",),
            "healthcare_ai",
            note="Imaging concept paraphrased; diagnosis domain term anchors BM25.",
            relevant_domain="diagnosis",
            style="semantic",
        ),
        Query(
            "Q06",
            "spotting rare unusual events in network traffic",
            ("anomaly_detection",),
            "cybersecurity",
            note="Anomaly detection expressed by meaning, not the exact phrase.",
            relevant_domain="traffic",
            style="semantic",
        ),
        Query(
            "Q07",
            "intrusion monitoring and malware campaign analysis on endpoints",
            ("cyber_threat",),
            "cybersecurity",
            relevant_domain="endpoint",
            style="lexical",
        ),
        Query(
            "Q08",
            "suspicious transaction screening for payment abuse",
            ("fraud_detection",),
            "finance_risk",
            relevant_domain="payment",
            style="lexical",
        ),
        Query(
            "Q09",
            "credit default probability estimation and risk scoring",
            ("risk_scoring",),
            "finance_risk",
            relevant_domain="credit",
            style="lexical",
        ),
        Query(
            "Q10",
            "predicting how much power renewable plants will produce",
            ("energy_forecasting",),
            "climate_energy",
            note="Energy forecasting paraphrased; renewable domain term anchors BM25.",
            relevant_domain="renewable",
            style="semantic",
        ),
        Query(
            "Q11",
            "seasonal temporal signal forecasting",
            ("time_series",),
            "climate_energy",
            relevant_domain="forecast",
            style="mixed",
        ),
        Query(
            "Q12",
            "personalized recommendation from a user profile graph",
            ("graph_recommendation",),
            "recommender_systems",
            relevant_domain="profile",
            style="mixed",
        ),
        Query(
            "Q13",
            "grouping the audience into customer cohorts from sessions",
            ("user_clustering",),
            "recommender_systems",
            note="Clustering paraphrased as grouping; session domain term anchors BM25.",
            relevant_domain="session",
            style="semantic",
        ),
        Query(
            "Q14",
            "autonomous path planning for warehouse robots",
            ("robot_navigation",),
            "robotics",
            relevant_domain="warehouse",
            style="lexical",
        ),
        Query(
            "Q15",
            "camera lidar fusion for perception",
            ("sensor_fusion",),
            "robotics",
            relevant_domain="camera",
            style="lexical",
        ),
        Query(
            "Q16",
            "genomic sequence matching across the genome",
            ("dna_alignment",),
            "bioinformatics",
            relevant_domain="genome",
            style="mixed",
        ),
        Query(
            "Q17",
            "case file tagging for contract documents",
            ("document_classification",),
            "legal_tech",
            relevant_domain="contract",
            style="lexical",
        ),
        Query(
            "Q18",
            "ordering the most relevant security alerts first",
            ("ranking",),
            "cybersecurity",
            note="Ranking paraphrased; alert domain term anchors BM25.",
            relevant_domain="alert",
            style="semantic",
        ),
        Query(
            "Q19",
            "outlier detection in patient records",
            ("anomaly_detection",),
            "healthcare_ai",
            relevant_domain="patient",
            style="mixed",
        ),
        Query(
            "Q20",
            "meaning based lookup over genomic sequence literature",
            ("semantic_search",),
            "bioinformatics",
            note="Semantic retrieval paraphrased; sequence domain term anchors BM25.",
            relevant_domain="sequence",
            style="semantic",
        ),
    ]


def documents_to_rows(documents: list[Document]) -> list[dict[str, object]]:
    return [document.to_dict() for document in documents]
