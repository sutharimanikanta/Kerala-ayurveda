# D:\Download1\kerala_ayurveda_content_pack_v1\product_knowledge.py
"""
Structured Product Knowledge Base
Converts product information into RAG-compatible chunks
"""

NALPAMARADI_PRODUCT_DATA = {
    "product_name": "Nalpamaradi Keram (Oil)",
    "brand": "Kerala Ayurveda",
    "rating": 4.5,
    "pricing": {
        "mrp": "₹180",
        "mrp_numeric": 180,
        "currency": "INR",
        "free_shipping_threshold": "₹299",
        "delivery_time": "2 to 4 days"
    },
    "origin": {
        "country": "India"
    },
    "description": {
        "primary": "Kerala Ayurveda Nalpamaradi Keram is a skin-brightening oil for the face and body. Enriched with the goodness of Nalpamara, it brightens dull skin, repairs pigmentation and environmental damage, promotes even skin tone, and aids natural glow.",
        "tagline": "Nourishes the skin, supports an even tone, and enhances natural radiance for sun-exposed areas."
    },
    "benefits": [
        "Helps reduce tanning and sun damage",
        "Brightens dull skin and enhances complexion",
        "Repairs pigmentation and environmental damage",
        "Improves skin tone and texture",
        "Promotes natural glow",
        "Nourishes the skin",
        "Supports even skin tone",
        "Enhances natural radiance for sun-exposed areas"
    ],
    "key_ingredients": [
        {
            "name": "Nalpamara",
            "sanskrit_name": "Nalpamara",
            "description": "Bark of four ficus trees",
            "benefits": "Improves skin health and natural complexion"
        },
        {
            "name": "Vetiver",
            "sanskrit_name": "Ushira",
            "description": "Vetiver root",
            "benefits": "Soothes and hydrates the skin"
        },
        {
            "name": "Indian Costus Root",
            "sanskrit_name": "Kushta",
            "description": "Costus root",
            "benefits": "Provides intensive skin repair"
        },
        {
            "name": "Indian Madder",
            "sanskrit_name": "Manjistha",
            "description": "Madder root",
            "benefits": "Helps improve skin complexion and reduces skin pigmentation"
        },
        {
            "name": "Turmeric",
            "sanskrit_name": "Haridra",
            "description": "Turmeric root",
            "benefits": "Reduces sun damage and promotes natural glow"
        }
    ],
    "usage_instructions": {
        "face": {
            "steps": [
                "Massage the oil all over your face and neck in gentle circular motions",
                "If you have dry skin, leave it for about 30 minutes",
                "If you have oily skin, leave it for about 10 minutes",
                "Apply the oil lightly if you have active acne, cuts, or wounds",
                "Afterwards, wash your face with warm water and a gentle cleanser"
            ],
            "duration_dry_skin": "30 minutes",
            "duration_oily_skin": "10 minutes",
            "precautions": "Apply lightly if you have active acne, cuts, or wounds"
        },
        "body": {
            "steps": [
                "Apply the required amount of oil to your body in long strokes about half an hour before the bath",
                "Take a bath in lukewarm water"
            ],
            "application_time": "30 minutes before bath",
            "water_temperature": "lukewarm"
        }
    },
    "product_category": "Ayurvedic Skin Care Oil",
    "application_areas": ["Face", "Body", "Neck"],
    "skin_concerns": [
        "Dull skin",
        "Pigmentation",
        "Uneven skin tone",
        "Sun damage",
        "Tanning",
        "Environmental damage",
        "Lack of radiance"
    ]
}


def convert_product_to_chunks():
    """
    Convert structured product data into RAG-compatible text chunks

    Returns:
        List of dictionaries with chunk data
    """
    chunks = []

    # Chunk 1: Product Overview
    overview_text = f"""Product Name: {NALPAMARADI_PRODUCT_DATA['product_name']}
Brand: {NALPAMARADI_PRODUCT_DATA['brand']}
Category: {NALPAMARADI_PRODUCT_DATA['product_category']}

Description: {NALPAMARADI_PRODUCT_DATA['description']['primary']}

Tagline: {NALPAMARADI_PRODUCT_DATA['description']['tagline']}

Country of Origin: {NALPAMARADI_PRODUCT_DATA['origin']['country']}"""

    chunks.append({
        'chunk_id': 'product_overview',
        'doc_id': 'nalpamaradi_product_info',
        'page': 1,
        'text': overview_text,
        'metadata': {
            'language': 'english',
            'corpus_type': 'product',
            'section': 'overview',
            'has_sanskrit': False,
            'word_count': len(overview_text.split())
        }
    })

    # Chunk 2: Pricing and Delivery
    pricing_text = f"""Pricing Information for {NALPAMARADI_PRODUCT_DATA['product_name']}:

M.R.P.: {NALPAMARADI_PRODUCT_DATA['pricing']['mrp']}
Free Shipping: Available on orders over {NALPAMARADI_PRODUCT_DATA['pricing']['free_shipping_threshold']}
Delivery Time: {NALPAMARADI_PRODUCT_DATA['pricing']['delivery_time']}

This product offers affordable pricing with quick delivery options for customers across India."""

    chunks.append({
        'chunk_id': 'product_pricing',
        'doc_id': 'nalpamaradi_product_info',
        'page': 1,
        'text': pricing_text,
        'metadata': {
            'language': 'english',
            'corpus_type': 'product',
            'section': 'pricing',
            'has_sanskrit': False,
            'word_count': len(pricing_text.split())
        }
    })

    # Chunk 3: Benefits
    benefits_text = f"""Benefits of {NALPAMARADI_PRODUCT_DATA['product_name']}:

"""
    for benefit in NALPAMARADI_PRODUCT_DATA['benefits']:
        benefits_text += f"• {benefit}\n"

    benefits_text += f"""
This Ayurvedic oil addresses multiple skin concerns including {', '.join(NALPAMARADI_PRODUCT_DATA['skin_concerns'][:3])}, and more."""

    chunks.append({
        'chunk_id': 'product_benefits',
        'doc_id': 'nalpamaradi_product_info',
        'page': 2,
        'text': benefits_text,
        'metadata': {
            'language': 'english',
            'corpus_type': 'product',
            'section': 'benefits',
            'has_sanskrit': False,
            'word_count': len(benefits_text.split())
        }
    })

    # Chunk 4: Key Ingredients (Detailed)
    ingredients_text = f"""Key Ingredients in {NALPAMARADI_PRODUCT_DATA['product_name']}:

"""
    for ingredient in NALPAMARADI_PRODUCT_DATA['key_ingredients']:
        ingredients_text += f"""
{ingredient['name']} ({ingredient['sanskrit_name']}):
Description: {ingredient['description']}
Benefits: {ingredient['benefits']}
"""

    chunks.append({
        'chunk_id': 'product_ingredients_detailed',
        'doc_id': 'nalpamaradi_product_info',
        'page': 2,
        'text': ingredients_text,
        'metadata': {
            'language': 'mixed',
            'corpus_type': 'product',
            'section': 'ingredients',
            'has_sanskrit': True,
            'word_count': len(ingredients_text.split())
        }
    })

    # Chunk 5: Ingredients Summary (for quick matching)
    ingredients_summary = f"""The main ingredients of {NALPAMARADI_PRODUCT_DATA['product_name']} are:

"""
    ingredient_names = [f"{ing['name']} (Sanskrit: {ing['sanskrit_name']})"
                       for ing in NALPAMARADI_PRODUCT_DATA['key_ingredients']]
    ingredients_summary += ", ".join(ingredient_names) + "."

    ingredients_summary += """

These traditional Ayurvedic herbs work synergistically to brighten skin, reduce pigmentation, and enhance natural radiance."""

    chunks.append({
        'chunk_id': 'product_ingredients_summary',
        'doc_id': 'nalpamaradi_product_info',
        'page': 2,
        'text': ingredients_summary,
        'metadata': {
            'language': 'mixed',
            'corpus_type': 'product',
            'section': 'ingredients',
            'has_sanskrit': True,
            'word_count': len(ingredients_summary.split())
        }
    })

    # Chunk 6: Usage for Face
    face_usage = f"""How to Use {NALPAMARADI_PRODUCT_DATA['product_name']} on Face:

"""
    for step in NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['steps']:
        face_usage += f"• {step}\n"

    face_usage += f"""
Timing Guidelines:
- Dry skin: Leave on for {NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['duration_dry_skin']}
- Oily skin: Leave on for {NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['duration_oily_skin']}

Precautions: {NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['precautions']}"""

    chunks.append({
        'chunk_id': 'product_usage_face',
        'doc_id': 'nalpamaradi_product_info',
        'page': 3,
        'text': face_usage,
        'metadata': {
            'language': 'english',
            'corpus_type': 'product',
            'section': 'usage',
            'has_sanskrit': False,
            'word_count': len(face_usage.split())
        }
    })

    # Chunk 7: Usage for Body
    body_usage = f"""How to Use {NALPAMARADI_PRODUCT_DATA['product_name']} on Body:

"""
    for step in NALPAMARADI_PRODUCT_DATA['usage_instructions']['body']['steps']:
        body_usage += f"• {step}\n"

    body_usage += f"""
Application Timing: Apply {NALPAMARADI_PRODUCT_DATA['usage_instructions']['body']['application_time']}
Water Temperature: Use {NALPAMARADI_PRODUCT_DATA['usage_instructions']['body']['water_temperature']} water for bathing

This traditional Ayurvedic practice allows the oil to penetrate deeply and nourish the skin."""

    chunks.append({
        'chunk_id': 'product_usage_body',
        'doc_id': 'nalpamaradi_product_info',
        'page': 3,
        'text': body_usage,
        'metadata': {
            'language': 'english',
            'corpus_type': 'product',
            'section': 'usage',
            'has_sanskrit': False,
            'word_count': len(body_usage.split())
        }
    })

    # Chunk 8: Skin Concerns Addressed
    concerns_text = f"""{NALPAMARADI_PRODUCT_DATA['product_name']} addresses the following skin concerns:

"""
    for concern in NALPAMARADI_PRODUCT_DATA['skin_concerns']:
        concerns_text += f"• {concern}\n"

    concerns_text += """
This makes it an ideal choice for those looking to improve their skin's overall health and appearance naturally through Ayurvedic principles."""

    chunks.append({
        'chunk_id': 'product_skin_concerns',
        'doc_id': 'nalpamaradi_product_info',
        'page': 3,
        'text': concerns_text,
        'metadata': {
            'language': 'english',
            'corpus_type': 'product',
            'section': 'concerns',
            'has_sanskrit': False,
            'word_count': len(concerns_text.split())
        }
    })

    return chunks


def get_product_faqs():
    """
    Generate FAQ-style chunks for common questions

    Returns:
        List of FAQ chunks
    """
    faqs = [
        {
            'question': 'What is Nalpamaradi Keram?',
            'answer': f"{NALPAMARADI_PRODUCT_DATA['description']['primary']}"
        },
        {
            'question': 'What are the main benefits of Nalpamaradi Keram?',
            'answer': f"The main benefits include: {', '.join(NALPAMARADI_PRODUCT_DATA['benefits'][:4])}."
        },
        {
            'question': 'What are the key ingredients in Nalpamaradi Keram?',
            'answer': f"The key ingredients are {', '.join([ing['name'] + ' (' + ing['sanskrit_name'] + ')' for ing in NALPAMARADI_PRODUCT_DATA['key_ingredients']])}."
        },
        {
            'question': 'How do I use Nalpamaradi Keram on my face?',
            'answer': ' '.join(NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['steps'][:3])
        },
        {
            'question': 'What is the price of Nalpamaradi Keram?',
            'answer': f"The M.R.P. is {NALPAMARADI_PRODUCT_DATA['pricing']['mrp']} with free shipping on orders over {NALPAMARADI_PRODUCT_DATA['pricing']['free_shipping_threshold']}."
        },
        {
            'question': 'How long should I leave Nalpamaradi Keram on my face?',
            'answer': f"For dry skin, leave it on for {NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['duration_dry_skin']}. For oily skin, leave it on for {NALPAMARADI_PRODUCT_DATA['usage_instructions']['face']['duration_oily_skin']}."
        },
        {
            'question': 'What skin concerns does Nalpamaradi Keram address?',
            'answer': f"It addresses {', '.join(NALPAMARADI_PRODUCT_DATA['skin_concerns'])}."
        }
    ]

    chunks = []
    for idx, faq in enumerate(faqs):
        faq_text = f"Q: {faq['question']}\nA: {faq['answer']}"

        chunks.append({
            'chunk_id': f'product_faq_{idx}',
            'doc_id': 'nalpamaradi_product_info',
            'page': 4,
            'text': faq_text,
            'metadata': {
                'language': 'english',
                'corpus_type': 'product',
                'section': 'faq',
                'has_sanskrit': 'sanskrit_name' in faq['answer'].lower(),
                'word_count': len(faq_text.split())
            }
        })

    return chunks


def get_all_product_chunks():
    """
    Get all product information chunks

    Returns:
        Combined list of all chunks
    """
    main_chunks = convert_product_to_chunks()
    faq_chunks = get_product_faqs()

    return main_chunks + faq_chunks