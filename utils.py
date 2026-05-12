# D:\Download1\kerala_ayurveda_content_pack_v1\utils.py
"""
Utility Functions for Nalpamaradi Keram System
"""

import re
from typing import List, Dict


def format_citations(citations: List[Dict]) -> str:
    """Format citations for display"""
    formatted = []
    for cit in citations:
        formatted.append(f"[{cit['label']}] {cit['doc_id']} (Page {cit['page']})")
    return "\n".join(formatted)


def validate_medical_claims(text: str) -> Dict:
    """
    Detect potential medical claims in text

    Returns dict with validation results
    """
    prohibited_patterns = {
        'high': [
            r'\b(cure|cures|cured|curing)\b',
            r'\b(treat|treats|treated|treating|treatment of)\b',
            r'\b(prevent|prevents|prevented|preventing|prevention of)\b',
            r'\b(diagnose|diagnosis)\b',
            r'\b(reverse|reverses|reversed)\b',
            r'\b(eliminate|eliminates|eliminated)\b'
        ],
        'medium': [
            r'\b(fix|fixes|fixed)\b',
            r'\b(heal|heals|healing)\b',
            r'\b(remedy for|remedies)\b'
        ]
    }

    disease_terms = [
        'disease', 'disorder', 'condition', 'syndrome', 'illness',
        'infection', 'cancer', 'diabetes', 'hypertension', 'arthritis',
        'eczema', 'psoriasis', 'acne', 'dermatitis'
    ]

    flagged_phrases = []
    severity = 'low'
    text_lower = text.lower()

    # Check prohibited patterns
    for sev, patterns in prohibited_patterns.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                flagged_phrases.append(context)

                if sev == 'high':
                    severity = 'high'
                elif sev == 'medium' and severity != 'high':
                    severity = 'medium'

    # Check disease terms with action verbs
    for disease in disease_terms:
        if disease in text_lower:
            disease_pattern = rf'\b{disease}\b'
            for match in re.finditer(disease_pattern, text_lower):
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text_lower[start:end]

                if any(re.search(p, context) for patterns in prohibited_patterns.values() for p in patterns):
                    flagged_phrases.append(text[start:end])
                    severity = 'high'

    return {
        'has_medical_claims': len(flagged_phrases) > 0,
        'flagged_phrases': flagged_phrases,
        'severity': severity,
        'count': len(flagged_phrases)
    }


def calculate_readability_score(text: str) -> Dict:
    """Calculate basic readability metrics"""
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    avg_sentence_length = word_count / max(sentence_count, 1)

    if avg_sentence_length < 15:
        level = 'Easy'
    elif avg_sentence_length < 20:
        level = 'Moderate'
    elif avg_sentence_length < 25:
        level = 'Challenging'
    else:
        level = 'Complex'

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'readability_level': level
    }


def check_tone_compliance(text: str) -> Dict:
    """Check if text follows Kerala Ayurveda tone guidelines"""
    issues = []
    positive_indicators = []
    text_lower = text.lower()

    # Negative indicators
    negative_patterns = {
        'Too clinical': [r'\b(clinical|therapeutic|pharmaceutical)\b'],
        'Too aggressive': [r'\b(must|always|never|definitely)\b'],
        'Overpromising': [r'\b(guarantee|guaranteed|proven|scientifically proven)\b'],
        'Too salesy': [r'\b(buy now|limited time|special offer|discount)\b']
    }

    for issue_type, patterns in negative_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                issues.append(issue_type)
                break

    # Positive indicators
    positive_patterns = [
        (r'\b(traditionally|classical|ancient)\b', 'Traditional framing'),
        (r'\b(may|might|can|support|help)\b', 'Gentle language'),
        (r'\b(consult|practitioner|guidance)\b', 'Professional guidance'),
        (r'\b(ritual|practice|routine)\b', 'Lifestyle integration'),
        (r'\b(natural|gentle|nourish)\b', 'Soft descriptors')
    ]

    for pattern, indicator in positive_patterns:
        if re.search(pattern, text_lower):
            positive_indicators.append(indicator)

    is_compliant = len(issues) == 0 and len(positive_indicators) >= 2

    return {
        'is_compliant': is_compliant,
        'issues': list(set(issues)),
        'positive_indicators': list(set(positive_indicators))
    }


def generate_summary_stats(draft: Dict, fact_check: Dict) -> Dict:
    """Generate comprehensive summary statistics"""
    total_paras = len(draft['paragraphs'])
    narrative_paras = sum(1 for p in draft['paragraphs'] if p.get('type') == 'narrative')
    factual_paras = total_paras - narrative_paras

    total_citations = sum(len(p.get('citations', [])) for p in draft['paragraphs'])
    paras_with_citations = sum(1 for p in draft['paragraphs'] if p.get('citations'))

    total_claims = sum(len(c.get('claims', [])) for c in fact_check.get('checks', []))
    supported_claims = sum(
        sum(1 for claim in c.get('claims', []) if claim.get('status') == 'supported')
        for c in fact_check.get('checks', [])
    )

    issues_by_severity = {
        'high': sum(1 for i in fact_check.get('issues', []) if i['severity'] == 'high'),
        'medium': sum(1 for i in fact_check.get('issues', []) if i['severity'] == 'medium'),
        'low': sum(1 for i in fact_check.get('issues', []) if i['severity'] == 'low')
    }

    full_text = "\n".join(p['text'] for p in draft['paragraphs'])
    readability = calculate_readability_score(full_text)
    tone_check = check_tone_compliance(full_text)

    return {
        'paragraph_stats': {
            'total': total_paras,
            'narrative': narrative_paras,
            'factual': factual_paras
        },
        'citation_stats': {
            'total_citations': total_citations,
            'paragraphs_with_citations': paras_with_citations,
            'citation_density': round(total_citations / max(total_paras, 1), 2)
        },
        'fact_check_stats': {
            'total_claims': total_claims,
            'supported_claims': supported_claims,
            'support_rate': round(supported_claims / max(total_claims, 1) * 100, 1),
            'issues_by_severity': issues_by_severity
        },
        'readability': readability,
        'tone_compliance': tone_check
    }