# vector_store.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "patient-vector")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

INPUT_JSON = os.getenv("INPUT_JSON", "patient_data.json")

def build_text_for_embedding(item: dict, item_type: str, item_id: str) -> str:
    """
    Construct a searchable text blob from patient support data.
    Different formats for different data types.
    """
    parts = []
    
    if item_type == "patient_education":
        parts.append(f"Disease: {item.get('disease_name', '')}")
        parts.append(f"Overview: {item.get('overview', '')}")
        
        if 'key_facts' in item:
            parts.append("Key Facts:")
            for fact in item['key_facts']:
                parts.append(f"  - {fact}")
        
        if 'medications' in item:
            parts.append("Medications:")
            for med in item['medications']:
                parts.append(f"  {med.get('name', '')}: {med.get('purpose', '')}")
                if 'common_brands' in med:
                    parts.append(f"    Brands: {', '.join(med['common_brands'])}")
                if 'important_notes' in med:
                    parts.append(f"    Important: {med['important_notes']}")
        
        if 'lifestyle_tips' in item:
            parts.append("Lifestyle Tips:")
            for tip in item['lifestyle_tips']:
                parts.append(f"  - {tip}")
        
        if 'when_to_seek_help' in item:
            parts.append("When to Seek Help:")
            for warning in item['when_to_seek_help']:
                parts.append(f"  - {warning}")
                
    elif item_type == "adherence_tools":
        parts.append(f"Adherence Topic: {item_id}")
        if 'tips' in item:
            parts.append("Tips:")
            for tip in item['tips']:
                parts.append(f"  - {tip}")
        if 'common_challenges' in item:
            parts.append("Common Challenges:")
            for challenge in item['common_challenges']:
                parts.append(f"  - {challenge}")
        if 'solutions' in item:
            parts.append("Solutions:")
            for solution in item['solutions']:
                parts.append(f"  - {solution}")
        if 'what_to_track' in item:
            parts.append("What to Track:")
            for item_track in item['what_to_track']:
                parts.append(f"  - {item_track}")
        if 'tracking_methods' in item:
            parts.append("Tracking Methods:")
            for method in item['tracking_methods']:
                parts.append(f"  - {method}")
        if 'strategies' in item:
            parts.append("Strategies:")
            for strategy in item['strategies']:
                parts.append(f"  - {strategy}")
                
    elif item_type == "symptom_tracking":
        parts.append(f"Symptom Tracking for: {item_id}")
        if 'symptoms_to_track' in item:
            parts.append("Symptoms to Track:")
            for symptom in item['symptoms_to_track']:
                parts.append(f"  - {symptom}")
        if 'tracking_frequency' in item:
            parts.append(f"Tracking Frequency: {item['tracking_frequency']}")
        if 'red_flags' in item:
            parts.append("Red Flags (Seek Immediate Help):")
            for flag in item['red_flags']:
                parts.append(f"  - {flag}")
                
    elif item_type == "patient_journey":
        parts.append(f"Patient Journey Stage: {item.get('stage', '')}")
        if 'typical_duration' in item:
            parts.append(f"Typical Duration: {item['typical_duration']}")
        if 'key_milestones' in item:
            parts.append("Key Milestones:")
            for milestone in item['key_milestones']:
                parts.append(f"  - {milestone}")
        if 'support_needed' in item:
            parts.append("Support Needed:")
            for support in item['support_needed']:
                parts.append(f"  - {support}")
                
    elif item_type == "support_programs":
        if 'programs' in item:
            parts.append("Support Programs:")
            for program in item['programs']:
                parts.append(f"  {program.get('name', '')}: {program.get('description', '')}")
                if 'benefits' in program:
                    parts.append("    Benefits:")
                    for benefit in program['benefits']:
                        parts.append(f"      - {benefit}")
                if 'duration' in program:
                    parts.append(f"    Duration: {program['duration']}")
                if 'eligibility' in program:
                    parts.append(f"    Eligibility: {program['eligibility']}")
        if 'online_resources' in item:
            parts.append("Online Resources:")
            for resource in item['online_resources']:
                parts.append(f"  - {resource}")
        if 'crisis_resources' in item:
            parts.append("Crisis Resources:")
            for resource in item['crisis_resources']:
                parts.append(f"  - {resource}")
    
    text_blob = "\n".join(parts).strip()
    return text_blob[:6000]  # Keep within reasonable size

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def get_aws_region(region_string: str):
    """Get AwsRegion enum value from string, with fallback."""
    region_map = {
        "us-east-1": AwsRegion.US_EAST_1,
        "us-west-2": AwsRegion.US_WEST_2,
        "eu-west-1": AwsRegion.EU_WEST_1,
    }
    
    if region_string in region_map:
        return region_map[region_string]
    
    try:
        enum_name = region_string.replace("-", "_").upper()
        if hasattr(AwsRegion, enum_name):
            return getattr(AwsRegion, enum_name)
    except Exception:
        pass
    
    print(f"⚠️  Region '{region_string}' not found, defaulting to us-east-1")
    return AwsRegion.US_EAST_1

def store_embeddings():
    """Store embeddings in Pinecone index."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get embedding dimension from OpenAI model
    print(f"Getting embedding dimension for model: {EMBEDDING_MODEL}")
    sample_embedding = create_embedding("sample")
    embedding_dimension = len(sample_embedding)
    print(f"✅ Embedding dimension: {embedding_dimension}")
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        
        aws_region = get_aws_region(PINECONE_REGION)
        
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=aws_region
            )
        )
        print(f"✅ Index {PINECONE_INDEX_NAME} created successfully with dimension {embedding_dimension}!")
    else:
        print(f"✅ Index {PINECONE_INDEX_NAME} already exists")
        index_stats = pc.describe_index(PINECONE_INDEX_NAME)
        index_dimension = index_stats.dimension
        
        if index_dimension != embedding_dimension:
            raise ValueError(
                f"❌ Dimension mismatch!\n"
                f"   Index '{PINECONE_INDEX_NAME}' has dimension: {index_dimension}\n"
                f"   Embedding model '{EMBEDDING_MODEL}' produces dimension: {embedding_dimension}\n\n"
                f"   Solutions:\n"
                f"   1. Use a different index name (set PINECONE_INDEX_NAME in .env)\n"
                f"   2. Delete the existing index and recreate it\n"
                f"   3. Use an embedding model that matches the index dimension\n"
            )
        else:
            print(f"✅ Index dimension ({index_dimension}) matches embedding dimension ({embedding_dimension})")
    
    # Connect to index
    index = pc.Index(name=PINECONE_INDEX_NAME)
    
    # Load JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Prepare vectors for batch upsert
    vectors_to_upsert = []
    
    doc_count = 0
    
    # Process Patient Education data
    if "patient_education" in data:
        print("\nProcessing Patient Education data...")
        for disease, info in data["patient_education"].items():
            uid = f"edu_{disease}"
            text = build_text_for_embedding(info, "patient_education", disease)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "patient_education",
                "disease": disease,
                "disease_name": info.get("disease_name", ""),
                "text": text
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    # Process Adherence Tools data
    if "adherence_tools" in data:
        print("\nProcessing Adherence Tools data...")
        for tool_type, tool_info in data["adherence_tools"].items():
            uid = f"adherence_{tool_type}"
            text = build_text_for_embedding(tool_info, "adherence_tools", tool_type)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "adherence_tools",
                "tool_type": tool_type,
                "text": text
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    # Process Symptom Tracking data
    if "symptom_tracking" in data:
        print("\nProcessing Symptom Tracking data...")
        if "common_symptoms" in data["symptom_tracking"]:
            for condition, symptom_info in data["symptom_tracking"]["common_symptoms"].items():
                uid = f"symptom_{condition}"
                text = build_text_for_embedding(symptom_info, "symptom_tracking", condition)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "symptom_tracking",
                    "condition": condition,
                    "text": text
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
                
                vectors_to_upsert.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                })
                doc_count += 1
    
    # Process Patient Journey data
    if "patient_journey" in data:
        print("\nProcessing Patient Journey data...")
        for journey_type, stages in data["patient_journey"].items():
            for stage_name, stage_info in stages.items():
                uid = f"journey_{journey_type}_{stage_name}"
                text = build_text_for_embedding(stage_info, "patient_journey", stage_name)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "patient_journey",
                    "journey_type": journey_type,
                    "stage": stage_info.get("stage", stage_name),
                    "text": text
                }
                metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
                
                vectors_to_upsert.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                })
                doc_count += 1
    
    # Process Support Programs data
    if "support_programs" in data:
        print("\nProcessing Support Programs data...")
        for program_type, program_info in data["support_programs"].items():
            uid = f"support_{program_type}"
            text = build_text_for_embedding(program_info, "support_programs", program_type)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "support_programs",
                "program_type": program_type,
                "text": text
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and v != ""}
            
            vectors_to_upsert.append({
                "id": uid,
                "values": embedding,
                "metadata": metadata
            })
            doc_count += 1
    
    print(f"\nTotal documents prepared: {doc_count}")
    
    # Batch upsert to Pinecone (upsert in batches of 100)
    batch_size = 100
    total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
    
    print(f"\nAdding documents to Pinecone in {total_batches} batches...")
    
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        batch_num = i // batch_size + 1
        print(f"   Upserted batch {batch_num}/{total_batches} ({len(batch)} documents)")
    
    print(f"\n✅ All embeddings stored successfully in Pinecone!")
    print(f"   Total documents: {doc_count}")

if __name__ == "__main__":
    store_embeddings()
