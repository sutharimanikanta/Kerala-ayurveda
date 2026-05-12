# D:\Download1\kerala_ayurveda_content_pack_v1\agents.py
"""
Specialized Agents for Content Generation Pipeline
Updated to use intent-aware retrieval architecture
"""

import re
import json
import logging
from typing import List, Dict

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutlineAgent:
    """
    Generates structured article outlines with RAG validation
    """

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        logger.info("✓ OutlineAgent initialized")

    def generate_outline(self, user_input: Dict) -> Dict:
        """
        Generate article outline based on brief

        Args:
            user_input: Dict with 'brief', 'product', 'audience', 'word_count', 'tone'

        Returns:
            Structured outline with sections
        """
        brief = user_input.get("brief", "")
        product = user_input.get("product", config.PRODUCT_NAME)
        audience = user_input.get("audience", "general consumers")
        word_count = user_input.get("word_count", 800)
        tone = user_input.get("tone", "warm and educational")

        # Use intent-aware retrieval for outline planning
        retrieval = self.rag.intent_aware_retrieve(
            query=brief,
            use_product_direct=True,  # Product-first for outline
        )

        # Build system prompt
        system_prompt = f"""You are an expert content planner for {config.BRAND_NAME}.

Create a structured article outline based on the brief.

Requirements:
- Target: {word_count} words
- Audience: {audience}
- Tone: {tone}
- Must include introduction, 3-5 body sections, and conclusion
- Each section should have a clear focus and key points
- Ensure content is grounded in available knowledge (product info and traditional texts)

Output ONLY valid JSON with this structure:
{{
  "title": "Article title",
  "sections": [
    {{
      "heading": "Section heading",
      "focus": "What this section covers",
      "key_points": ["point 1", "point 2"],
      "estimated_words": 150
    }}
  ]
}}

NO markdown, NO preamble, ONLY JSON."""

        # Build prompt with separated contexts
        prompt = self.rag.build_prompt(
            query=brief,
            product_chunks=retrieval["product_chunks"],
            pdf_chunks=retrieval["pdf_chunks"],
            system_prompt=system_prompt,
        )

        # Generate outline
        response = self.rag.call_llm(
            prompt, temperature=config.OUTLINE_TEMPERATURE, max_tokens=1500
        )

        # Parse JSON response
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            outline = json.loads(response)

            # Validate structure
            if "sections" not in outline:
                raise ValueError("Missing 'sections' in outline")

            logger.info(f"✓ Outline generated: {len(outline['sections'])} sections")

            return outline

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse outline JSON: {e}")
            logger.error(f"Raw response: {response[:500]}")
            raise ValueError(f"Invalid JSON from outline generation: {e}")


class WriterAgent:
    """
    Generates article drafts from outlines with citations
    """

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        logger.info("✓ WriterAgent initialized")

    def generate_draft(self, outline: Dict) -> Dict:
        """
        Generate full article draft from outline

        Args:
            outline: Structured outline from OutlineAgent

        Returns:
            Draft with paragraphs and citations
        """
        sections = outline.get("sections", [])
        paragraphs = []

        for section in sections:
            # Query for this section
            section_query = f"{section['heading']}: {section['focus']}"

            # Retrieve relevant content with intent-aware retrieval
            retrieval = self.rag.intent_aware_retrieve(
                query=section_query, use_product_direct=True
            )

            # Build system prompt for writing
            system_prompt = f"""You are an expert content writer for {config.BRAND_NAME}.

Write a detailed paragraph for this section:
- Heading: {section["heading"]}
- Focus: {section["focus"]}
- Key points to cover: {", ".join(section.get("key_points", []))}
- Target length: {section.get("estimated_words", 150)} words

Requirements:
- Use warm, educational tone
- Ground all claims in provided context
- Cite sources using [P1], [P2] for product info and [T1], [T2] for traditional texts
- Never make unsupported medical claims
- Use phrases like "may support", "traditionally used for", "can help with"
- Make content engaging and readable

Write ONLY the paragraph text, no headings or labels."""

            # Build prompt with separated contexts
            prompt = self.rag.build_prompt(
                query=section_query,
                product_chunks=retrieval["product_chunks"],
                pdf_chunks=retrieval["pdf_chunks"],
                system_prompt=system_prompt,
            )

            # Generate paragraph
            paragraph_text = self.rag.call_llm(
                prompt, temperature=config.WRITER_TEMPERATURE, max_tokens=500
            )

            # Extract citations
            citations = self.rag.extract_citations(
                paragraph_text, retrieval["product_chunks"], retrieval["pdf_chunks"]
            )

            paragraphs.append(
                {
                    "section_heading": section["heading"],
                    "text": paragraph_text.strip(),
                    "citations": citations,
                    "type": "factual" if citations else "narrative",
                }
            )

        logger.info(f"✓ Draft generated: {len(paragraphs)} paragraphs")

        return {
            "title": outline.get("title", "Untitled"),
            "paragraphs": paragraphs,
            "outline": outline,
        }

    def revise_draft(
        self, draft: Dict, fact_check_issues: List[Dict], outline: Dict
    ) -> Dict:
        """
        Revise draft based on fact-checker feedback

        Args:
            draft: Original draft
            fact_check_issues: Issues from FactCheckerAgent
            outline: Original outline

        Returns:
            Revised draft
        """
        # Group issues by paragraph
        issues_by_para = {}
        for issue in fact_check_issues:
            para_idx = issue.get("paragraph_index", 0)
            if para_idx not in issues_by_para:
                issues_by_para[para_idx] = []
            issues_by_para[para_idx].append(issue)

        # Revise problematic paragraphs
        revised_paragraphs = []

        for i, paragraph in enumerate(draft["paragraphs"]):
            if i in issues_by_para:
                # Needs revision
                issues = issues_by_para[i]
                logger.info(f"Revising paragraph {i + 1}: {len(issues)} issues")

                # Build revision query
                section_heading = paragraph["section_heading"]
                issues_text = "\n".join(
                    [f"- {issue['type']}: {issue['message']}" for issue in issues]
                )

                revision_query = f"{section_heading} - addressing: {issues_text}"

                # Retrieve with intent-aware retrieval
                retrieval = self.rag.intent_aware_retrieve(
                    query=revision_query, use_product_direct=True
                )

                # Build revision prompt
                system_prompt = f"""You are revising content for {config.BRAND_NAME}.

Original paragraph:
{paragraph["text"]}

Issues to fix:
{issues_text}

Requirements:
- Fix all identified issues
- Maintain warm, educational tone
- Ground claims in provided context
- Use [P1], [P2] for product, [T1], [T2] for traditional texts
- Avoid unsupported medical claims
- Keep similar length and structure

Write ONLY the revised paragraph, no explanations."""

                # Build prompt
                prompt = self.rag.build_prompt(
                    query=revision_query,
                    product_chunks=retrieval["product_chunks"],
                    pdf_chunks=retrieval["pdf_chunks"],
                    system_prompt=system_prompt,
                )

                # Generate revision
                revised_text = self.rag.call_llm(
                    prompt, temperature=config.WRITER_TEMPERATURE, max_tokens=500
                )

                # Extract citations
                citations = self.rag.extract_citations(
                    revised_text, retrieval["product_chunks"], retrieval["pdf_chunks"]
                )

                revised_paragraphs.append(
                    {
                        "section_heading": section_heading,
                        "text": revised_text.strip(),
                        "citations": citations,
                        "type": "factual" if citations else "narrative",
                    }
                )

            else:
                # Keep original
                revised_paragraphs.append(paragraph)

        logger.info(f"✓ Draft revised: {len(issues_by_para)} paragraphs updated")

        return {
            "title": draft["title"],
            "paragraphs": revised_paragraphs,
            "outline": outline,
        }


class FactCheckerAgent:
    """
    Validates claims against knowledge base
    """

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        logger.info("✓ FactCheckerAgent initialized")

    def check_draft(self, draft: Dict) -> Dict:
        """
        Check all claims in draft against knowledge base

        Args:
            draft: Draft from WriterAgent

        Returns:
            Fact-check results with issues
        """
        checks = []
        all_issues = []

        for i, paragraph in enumerate(draft["paragraphs"]):
            para_text = paragraph["text"]

            # Skip narrative paragraphs (no factual claims)
            if paragraph.get("type") == "narrative":
                continue

            # Extract claims from paragraph
            claims = self._extract_claims(para_text)

            if not claims:
                continue

            # Verify each claim
            para_issues = []

            for claim in claims:
                # Query knowledge base for verification
                retrieval = self.rag.intent_aware_retrieve(
                    query=claim, use_product_direct=True
                )

                # Check if claim is supported
                is_supported = self._verify_claim(
                    claim, retrieval["product_chunks"], retrieval["pdf_chunks"]
                )

                if not is_supported["supported"]:
                    issue = {
                        "paragraph_index": i,
                        "claim": claim,
                        "type": "unsupported_claim",
                        "severity": is_supported["severity"],
                        "message": is_supported["reason"],
                        "suggestion": is_supported.get("suggestion", ""),
                    }
                    para_issues.append(issue)
                    all_issues.append(issue)

            checks.append(
                {
                    "paragraph_index": i,
                    "text": para_text,
                    "claims": claims,
                    "issues": para_issues,
                }
            )

        # Check for medical claims
        medical_issues = self._check_medical_claims(draft)
        all_issues.extend(medical_issues)

        logger.info(f"✓ Fact-check complete: {len(all_issues)} issues found")

        return {
            "checks": checks,
            "issues": all_issues,
            "total_issues": len(all_issues),
            "high_severity": sum(1 for i in all_issues if i["severity"] == "high"),
        }

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Filter for factual statements (contains specific keywords)
        factual_keywords = [
            "contains",
            "includes",
            "provides",
            "helps",
            "supports",
            "reduces",
            "improves",
            "enhances",
            "promotes",
            "aids",
            "known for",
            "traditionally used",
            "according to",
        ]

        claims = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in factual_keywords):
                claims.append(sentence)

        return claims

    def _verify_claim(
        self, claim: str, product_chunks: List[Dict], pdf_chunks: List[Dict]
    ) -> Dict:
        """Verify if claim is supported by knowledge base"""

        # Combine all available context
        all_text = " ".join([c["text"] for c in product_chunks + pdf_chunks])

        # Simple keyword overlap check
        claim_words = set(claim.lower().split())
        context_words = set(all_text.lower().split())

        overlap = len(claim_words & context_words)
        overlap_ratio = overlap / max(len(claim_words), 1)

        # Check for prohibited medical claims
        prohibited_patterns = config.MEDICAL_CLAIM_PATTERNS
        has_medical_claim = any(
            re.search(pattern, claim.lower()) for pattern in prohibited_patterns
        )

        if has_medical_claim:
            return {
                "supported": False,
                "severity": "high",
                "reason": "Contains prohibited medical claim",
                "suggestion": 'Rewrite using softer language like "may support" or "traditionally used for"',
            }

        if overlap_ratio < 0.3:
            return {
                "supported": False,
                "severity": "medium",
                "reason": "Claim has insufficient support in knowledge base",
                "suggestion": "Either provide citation or remove/soften claim",
            }

        return {
            "supported": True,
            "severity": "low",
            "reason": "Claim appears supported",
        }

    def _check_medical_claims(self, draft: Dict) -> List[Dict]:
        """Check entire draft for prohibited medical claims"""
        issues = []

        full_text = " ".join([p["text"] for p in draft["paragraphs"]])

        prohibited_patterns = config.PROHIBITED_MEDICAL_CLAIMS

        for pattern in prohibited_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # Find which paragraph
                para_idx = 0
                char_count = 0
                for i, para in enumerate(draft["paragraphs"]):
                    if char_count + len(para["text"]) > match.start():
                        para_idx = i
                        break
                    char_count += len(para["text"]) + 1

                issues.append(
                    {
                        "paragraph_index": para_idx,
                        "claim": match.group(),
                        "type": "prohibited_medical_claim",
                        "severity": "high",
                        "message": f'Prohibited medical language: "{match.group()}"',
                        "suggestion": 'Use softer language: "may support", "traditionally used for", etc.',
                    }
                )

        return issues


class FinalizationAgent:
    """
    Polishes and prepares final output
    """

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        logger.info("✓ FinalizationAgent initialized")

    def finalize(self, draft: Dict, fact_check: Dict) -> Dict:
        """
        Prepare final article output

        Args:
            draft: Final draft from WriterAgent
            fact_check: Results from FactCheckerAgent

        Returns:
            Finalized article ready for publication
        """
        # Assemble full text
        full_text = f"# {draft['title']}\n\n"

        for paragraph in draft["paragraphs"]:
            full_text += f"{paragraph['text']}\n\n"

        # Collect all citations
        all_citations = []
        seen = set()

        for paragraph in draft["paragraphs"]:
            for citation in paragraph.get("citations", []):
                cite_key = (
                    f"{citation['label']}_{citation['doc_id']}_{citation['page']}"
                )
                if cite_key not in seen:
                    all_citations.append(citation)
                    seen.add(cite_key)

        # Sort citations
        all_citations.sort(key=lambda x: x["label"])

        # Generate editor notes
        notes = []

        if fact_check["total_issues"] > 0:
            notes.append(
                f"⚠️ {fact_check['total_issues']} minor issues remaining "
                f"({fact_check['high_severity']} high severity). Review recommended."
            )

        if len(all_citations) == 0:
            notes.append("ℹ️ No citations - content is primarily narrative/introductory")

        if len(all_citations) > 10:
            notes.append(
                "ℹ️ High citation density - consider consolidating some references"
            )

        logger.info(
            f"✓ Article finalized: {len(all_citations)} citations, {len(notes)} notes"
        )

        return {
            "text": full_text.strip(),
            "citations": all_citations,
            "status": "ready_for_review"
            if fact_check["high_severity"] == 0
            else "needs_revision",
            "notes_for_editor": notes,
            "fact_check_summary": {
                "total_issues": fact_check["total_issues"],
                "high_severity": fact_check["high_severity"],
            },
        }
