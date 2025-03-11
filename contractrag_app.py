import os
import getpass
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from fuzzywuzzy import process
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import re
import datetime

# Define base path to the project
BASE_PATH = os.path.expanduser("~/Documents/contractrag_app")

# Set up Mistral API key automatically
os.environ["MISTRAL_API_KEY"] = "0OYFp5b31qOfBEYVppnkUlDmn4uuLen4"

# Initialize AI components
print("Initializing AI components...")
llm = ChatMistralAI(
    model="mistral-large-latest",
    timeout=120.0  # Increased timeout to 120 seconds
)
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embeddings)

# Define mappings for user input to categories
region_mapping = {
    # North America
    "usa": "North America",
    "united states": "North America",
    "canada": "North America",
    "mexico": "North America",
    
    # European Union
    "germany": "European Union",
    "france": "European Union",
    "spain": "European Union",
    "italy": "European Union",
    
    # United Kingdom
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    
    # Asia Pacific
    "australia": "Asia Pacific",
    "japan": "Asia Pacific",
    "singapore": "Asia Pacific",
    "china": "Asia Pacific",
    "india": "Asia Pacific",
    
    # Latin America (new addition)
    "brazil": "Latin America",
    "mexico": "Latin America",  # Note: Mexico appears in both North America and Latin America
    "argentina": "Latin America",
    "colombia": "Latin America",
    "chile": "Latin America",
    "peru": "Latin America",
    "venezuela": "Latin America",
    "ecuador": "Latin America",
    "bolivia": "Latin America",
    "uruguay": "Latin America",
    "paraguay": "Latin America",
    "costa rica": "Latin America",
    "panama": "Latin America",
    "latin america": "Latin America",
}

# Function to map user input to predefined categories
def map_to_region(user_input):
    normalized_input = user_input.lower().strip()
    
    # Try exact match first
    if normalized_input in region_mapping:
        return region_mapping[normalized_input]
    
    # Try fuzzy matching
    matches = process.extractOne(normalized_input, region_mapping.keys())
    if matches and matches[1] >= 80:  # 80% similarity threshold
        return region_mapping[matches[0]]
    
    # Default option if no match
    print("Region not specifically recognized. Defaulting to North America.")
    return "North America"

# Load the contract documents
def load_contract_documents():
    print("Loading contract documentation...")
    
    # Load regional standards documents
    regional_loader = DirectoryLoader(
        os.path.join(BASE_PATH, "ContractDocuments/RegionalStandards/"), 
        glob="**/*.md", 
        loader_cls=TextLoader
    )
    regional_docs = regional_loader.load()
    
    # Load company size documents
    size_loader = DirectoryLoader(
        os.path.join(BASE_PATH, "ContractDocuments/CompanySize/"), 
        glob="**/*.md", 
        loader_cls=TextLoader
    )
    size_docs = size_loader.load()
    
    # Add metadata to documents
    all_docs = []
    
    for doc in regional_docs:
        if "NorthAmerica" in doc.metadata["source"]:
            doc.metadata["region"] = "North America"
        elif "EuropeanUnion" in doc.metadata["source"]:
            doc.metadata["region"] = "European Union"
        elif "UnitedKingdom" in doc.metadata["source"]:
            doc.metadata["region"] = "United Kingdom"
        elif "AsiaPacific" in doc.metadata["source"]:
            doc.metadata["region"] = "Asia Pacific"
        elif "LatinAmerica" in doc.metadata["source"]:
            doc.metadata["region"] = "Latin America"
        
        doc.metadata["doc_type"] = "regional_standard"
        all_docs.append(doc)
    
    for doc in size_docs:
        if "VerySmall" in doc.metadata["source"]:
            doc.metadata["company_size"] = "Very Small"
        elif "Small" in doc.metadata["source"]:
            doc.metadata["company_size"] = "Small"
        elif "Medium" in doc.metadata["source"]:
            doc.metadata["company_size"] = "Medium"
        elif "Large" in doc.metadata["source"]:
            doc.metadata["company_size"] = "Large"
        elif "VeryLarge" in doc.metadata["source"]:
            doc.metadata["company_size"] = "Very Large"
        
        doc.metadata["doc_type"] = "company_size"
        all_docs.append(doc)
    
    return all_docs

# Split documents
def process_documents(documents):
    print("Processing documents...")
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} chunks")
    
    # Store in vector database
    vector_store.add_documents(documents=splits)
    print("Documents indexed successfully")

# Create custom prompt for contract analysis
contract_analysis_template = """You are a digital contract advisor for freelancers and contractors. 
Use the following information to provide a detailed contract analysis for the user's situation.

User Contract Profile:
- Region: {region}
- Project Difficulty: {difficulty}
- Client Company Size: {company_size}
- Hours Needed: {hours} hours
- Hourly Rate: ${hourly_rate}
- Payment Terms: {payment_terms}
- Revision Rounds: {revision_rounds} rounds
- Late Delivery Penalty: {late_penalty}%

Context from contract documentation:
{context}

Based on the user profile and the provided documentation, create a comprehensive contract analysis covering:

1. EXECUTIVE SUMMARY:
   - Overall assessment of the contract terms
   - Key strengths and weaknesses
   - Earning potential analysis

2. RATE ANALYSIS:
   - Comparison of user's hourly rate against benchmarks
   - Earning potential calculation showing how much more the user could make
   - Specific recommendations for rate optimization

3. PROJECT SCOPE ASSESSMENT:
   - Analysis of hours vs. project difficulty
   - Recommendations for scope management
   - Risk assessment based on revision rounds

4. PAYMENT STRUCTURE EVALUATION:
   - Analysis of payment terms compared to standards
   - Recommendations for better payment security
   - Cash flow implications

5. RISK MITIGATION:
   - Analysis of late delivery penalties
   - Recommendations for contract protections
   - Region-specific considerations

Use a direct, conversational tone as if you're a professional speaking to a colleague.
Avoid overly formal language, jargon, and passive voice.
Use contractions and be straightforward with recommendations.
Keep paragraphs focused on a single idea.
Emphasize practical, actionable advice and real earning potential opportunities.

Contract Analysis:"""

contract_analysis_prompt = PromptTemplate.from_template(contract_analysis_template)

# Define the state for our application
class State(TypedDict):
    profile: dict
    context: List[Document]
    analysis: str

def retrieve(state: State):
    # Extract criteria from the profile
    region = state["profile"]["region"]
    company_size = state["profile"]["company_size"]
    
    print(f"Retrieving relevant contract information for {region} and {company_size}...")
    
    # Create a query based on the user profile
    difficulty = state["profile"]["difficulty"]
    hourly_rate = state["profile"]["hourly_rate"]
    
    query = f"Contract standards for {region} region with {company_size} companies for {difficulty} difficulty projects at ${hourly_rate} hourly rate"
    
    # Retrieve documents matching the region
    region_docs = vector_store.similarity_search(
        query,
        filter=lambda doc: doc.metadata.get("region") == region,
        k=2
    )
    
    # Retrieve documents matching the company size
    size_docs = vector_store.similarity_search(
        query,
        filter=lambda doc: doc.metadata.get("company_size") == company_size,
        k=2
    )
    
    # Combine the results
    relevant_docs = region_docs + size_docs
    
    return {"context": relevant_docs}

def generate_fallback_analysis(profile):
    """Generate a basic analysis when the API call fails"""
    region = profile["region"]
    difficulty = profile["difficulty"]
    company_size = profile["company_size"]
    hourly_rate = profile["hourly_rate"]
    hours = profile["hours"]
    payment_terms = profile["payment_terms"]
    
    analysis = f"""
EXECUTIVE SUMMARY:
Your contract terms for this {difficulty} project with a {company_size} company in {region} show some strengths but also areas for improvement. We've analyzed your rate, scope, and payment structure to identify opportunities to optimize your agreement.

RATE ANALYSIS:
Your rate of ${hourly_rate}/hour is {'competitive' if hourly_rate > 60 else 'below standard market rates'} for {difficulty} projects in your region. Based on our data, similar contractors typically charge ${int(hourly_rate * 1.2)}-${int(hourly_rate * 1.4)}/hour for comparable work. This suggests a potential earning increase of ${int(hours * (hourly_rate * 0.3))} for this project.

PROJECT SCOPE ASSESSMENT:
A {hours}-hour estimate for a {difficulty} project seems {'appropriate' if difficulty == 'Easy' or difficulty == 'Medium' else 'potentially underestimated'}. Consider adding a buffer of 15-20% to account for unforeseen complexities.

PAYMENT STRUCTURE EVALUATION:
Your payment terms ({payment_terms}) are {'favorable' if '50%' in payment_terms or 'upfront' in payment_terms.lower() else 'standard but could be improved'}. We recommend negotiating for at least 40% upfront payment to improve cash flow and reduce risk.

RISK MITIGATION:
Include clear deliverable specifications, a change request process, and explicit approval criteria to protect against scope creep and dispute issues common in your region.
"""
    return analysis

def generate(state: State):
    # Format retrieved documents for the prompt
    docs_content = "\n\n".join([
        f"[{doc.metadata.get('doc_type', 'Document')} - {doc.metadata.get('region', '')} {doc.metadata.get('company_size', '')}]\n{doc.page_content}"
        for doc in state["context"]
    ])
    
    print("Generating contract analysis...")
    
    try:
        # Generate the analysis
        messages = contract_analysis_prompt.invoke({
            "region": state["profile"]["region"],
            "difficulty": state["profile"]["difficulty"],
            "company_size": state["profile"]["company_size"],
            "hours": state["profile"]["hours"],
            "hourly_rate": state["profile"]["hourly_rate"],
            "payment_terms": state["profile"]["payment_terms"],
            "revision_rounds": state["profile"]["revision_rounds"],
            "late_penalty": state["profile"]["late_penalty"],
            "context": docs_content
        })
        response = llm.invoke(messages)
        return {"analysis": response.content}
    except Exception as e:
        print(f"Error generating analysis: {e}")
        # Provide a fallback response
        return {"analysis": generate_fallback_analysis(state["profile"])}

# Build and compile the graph
def build_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()

# Function to create a professionally formatted PDF report
def create_pdf_report(profile, analysis, filename):
    # Create a PDF document
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        leftMargin=42, 
        rightMargin=42,
        topMargin=36,
        bottomMargin=36
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    # Create professional custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=14,
        fontName='Helvetica-Bold',
        leading=16,
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=12,
        fontName='Helvetica-Bold',
        leading=14,
        spaceBefore=8,
        spaceAfter=4
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitleStyle',
        parent=styles['Heading3'],
        fontSize=10,
        fontName='Helvetica-Bold',
        leading=12,
        spaceBefore=6,
        spaceAfter=2
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica',
        leading=11,
        spaceAfter=3
    )
    
    # List to hold the PDF elements
    elements = []
    
    # Add title
    elements.append(Paragraph("Contract Analysis Report", title_style))
    
    # Add date in smaller text
    date_str = datetime.datetime.now().strftime("%B %d, %Y")
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER
    )
    elements.append(Paragraph(f"Generated: {date_str}", date_style))
    elements.append(Spacer(1, 8))
    
    # Create a table for the profile
    profile_data = [
        ["Project Parameters", "Value"],
        ["Hours", f"{profile['hours']} hours"],
        ["Hourly Rate", f"${profile['hourly_rate']}"],
        ["Payment Terms", profile['payment_terms']],
        ["Revision Rounds", f"{profile['revision_rounds']}"],
        ["Late Penalty", f"{profile['late_penalty']}%"],
        ["Difficulty", profile['difficulty']],
        ["Region", profile['region']],
        ["Client Size", profile['company_size']]
    ]
    
    profile_table = Table(profile_data, colWidths=[120, 330])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (1, 0), 4),
        ('TOPPADDING', (0, 0), (1, 0), 4),
        
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    elements.append(profile_table)
    elements.append(Spacer(1, 12))
    
    # Process the analysis content for a more natural, professional tone
    # Clean up the analysis text - remove markdown symbols and extra whitespace
    cleaned_analysis = analysis.replace("#", "").replace("*", "")
    cleaned_analysis = re.sub(r'\s+', ' ', cleaned_analysis)
    
    # Define section patterns and titles
    section_patterns = [
        (r"(?i)EXECUTIVE\s+SUMMARY[\s:]+(.+?)(?=RATE ANALYSIS|$)", "Key Findings"),
        (r"(?i)RATE\s+ANALYSIS[\s:]+(.+?)(?=PROJECT SCOPE|$)", "Compensation Assessment"),
        (r"(?i)PROJECT\s+SCOPE(?:\s+ASSESSMENT)?[\s:]+(.+?)(?=PAYMENT STRUCTURE|$)", "Project Parameters"),
        (r"(?i)PAYMENT\s+STRUCTURE(?:\s+EVALUATION)?[\s:]+(.+?)(?=RISK MITIGATION|$)", "Payment Considerations"),
        (r"(?i)RISK\s+MITIGATION[\s:]+(.+?)(?=$)", "Risk Management")
    ]
    
    # Add analysis with improved language
    elements.append(Paragraph("Professional Recommendations", subtitle_style))
    
    found_sections = False
    
    # Function to improve text readability and professionalism
    def improve_text(text):
        # Replace overly formal phrases
        text = re.sub(r'it is recommended that', 'we recommend', text, flags=re.I)
        text = re.sub(r'it is advised that', 'we advise', text, flags=re.I)
        text = re.sub(r'it is suggested that', 'we suggest', text, flags=re.I)
        
        # Replace passive voice with active voice where possible
        text = re.sub(r'should be considered', 'consider', text, flags=re.I)
        text = re.sub(r'could be implemented', 'consider implementing', text, flags=re.I)
        text = re.sub(r'must be included', 'include', text, flags=re.I)
        
        # Make language more direct
        text = re.sub(r'in order to', 'to', text, flags=re.I)
        text = re.sub(r'for the purpose of', 'for', text, flags=re.I)
        text = re.sub(r'in the event that', 'if', text, flags=re.I)
        
        # Improve transitions
        text = re.sub(r'Additionally,', 'Also,', text, flags=re.I)
        text = re.sub(r'Furthermore,', 'Moreover,', text, flags=re.I)
        text = re.sub(r'Consequently,', 'As a result,', text, flags=re.I)
        
        # Remove redundant phrases
        text = re.sub(r'and thus,?', 'and', text, flags=re.I)
        text = re.sub(r'as a matter of fact', '', text, flags=re.I)
        
        # Simplify business jargon
        text = re.sub(r'utilize', 'use', text, flags=re.I)
        text = re.sub(r'facilitate', 'help', text, flags=re.I)
        text = re.sub(r'leverage', 'use', text, flags=re.I)
        
        return text
    
    for pattern, section_title in section_patterns:
        match = re.search(pattern, cleaned_analysis, re.DOTALL)
        if match:
            found_sections = True
            content = match.group(1).strip()
            
            elements.append(Paragraph(section_title, section_title_style))
            
            # Process the content into readable paragraphs
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Group sentences into logical paragraphs
            paragraphs = []
            current_para = []
            topic_indicators = ['rate', 'hour', 'project', 'payment', 'risk', 'recommend', 'consider']
            
            for i, sentence in enumerate(sentences):
                if len(current_para) == 0:
                    current_para.append(sentence)
                elif i > 0 and any(indicator in sentence.lower() for indicator in topic_indicators) and not any(indicator in sentences[i-1].lower() for indicator in topic_indicators):
                    # New topic likely starting - create a new paragraph
                    paragraphs.append(' '.join(current_para))
                    current_para = [sentence]
                elif len(' '.join(current_para)) + len(sentence) > 250:
                    # Current paragraph getting too long
                    paragraphs.append(' '.join(current_para))
                    current_para = [sentence]
                else:
                    current_para.append(sentence)
            
            if current_para:
                paragraphs.append(' '.join(current_para))
            
            # Improve the language of each paragraph and add to document
            for para in paragraphs:
                improved_para = improve_text(para)
                elements.append(Paragraph(improved_para, body_style))
    
    # If no sections were found, process the whole text
    if not found_sections:
        # Split by paragraphs and improve each one
        paragraphs = cleaned_analysis.split('\n\n')
        for para in paragraphs:
            if para.strip():
                improved_para = improve_text(para.strip())
                elements.append(Paragraph(improved_para, body_style))
    
    # Build the PDF
    doc.build(elements)
    print(f"PDF report saved as {filename}")

# Function to gather user responses through the questionnaire
def collect_user_responses():
    profile = {}
    print("\n" + "="*50)
    print("Digital Contract Advisor")
    print("="*50)
    print("\nPlease answer the following questions to receive a personalized contract analysis.\n")
    
    # Collect numerical variables
    print("PART 1: Numerical Contract Details")
    print("---------------------------------")
    try:
        profile["hours"] = int(input("\nHow many hours will the project take? "))
        profile["hourly_rate"] = float(input("\nWhat is your hourly rate in USD? $"))
        
        print("\nWhat are the payment terms?")
        print("  Examples: Net-30, 50% upfront, etc.")
        profile["payment_terms"] = input("\nYour answer: ")
        
        profile["revision_rounds"] = int(input("\nHow many revision rounds are included in the contract? "))
        profile["late_penalty"] = float(input("\nWhat is the late delivery penalty percentage? (Enter 0 if none) "))
    except ValueError:
        print("Please enter a valid number. Starting over...")
        return collect_user_responses()
    
    # Collect qualitative variables
    print("\nPART 2: Project and Client Details")
    print("---------------------------------")
    
    print("\nWhat is the project difficulty level?")
    print("  1. Easy")
    print("  2. Medium")
    print("  3. Hard")
    print("  4. Very Hard")
    
    try:
        difficulty_choice = int(input("\nEnter your choice (1-4): "))
        if difficulty_choice < 1 or difficulty_choice > 4:
            raise ValueError("Invalid selection")
            
        difficulty_options = ["Easy", "Medium", "Hard", "Very Hard"]
        profile["difficulty"] = difficulty_options[difficulty_choice-1]
    except (ValueError, IndexError):
        print("Please select a valid option. Starting over...")
        return collect_user_responses()
    
    print("\nWhat region are you operating in?")
    print("  Examples: USA, Canada, United Kingdom, Germany, Brazil, Mexico, Japan, etc.")
    print("  (North America, European Union, United Kingdom, Asia Pacific, and Latin America regions are supported)")
    region_input = input("\nYour answer: ")
    profile["region"] = map_to_region(region_input)
    print(f"  → Mapped to region: {profile['region']}")
    
    print("\nWhat is the client company size (annual revenue)?")
    print("  1. Very Small ($0-$50k)")
    print("  2. Small ($50k-$150k)")
    print("  3. Medium ($150k-$250k)")
    print("  4. Large ($250k-$500k)")
    print("  5. Very Large ($500k+)")
    
    try:
        size_choice = int(input("\nEnter your choice (1-5): "))
        if size_choice < 1 or size_choice > 5:
            raise ValueError("Invalid selection")
            
        size_options = ["Very Small", "Small", "Medium", "Large", "Very Large"]
        profile["company_size"] = size_options[size_choice-1]
    except (ValueError, IndexError):
        print("Please select a valid option. Starting over...")
        return collect_user_responses()
    
    # Show summary of collected information
    print("\nSummary of your contract details:")
    print("---------------------------------")
    print(f"• Hours needed: {profile['hours']}")
    print(f"• Hourly rate: ${profile['hourly_rate']}")
    print(f"• Payment terms: {profile['payment_terms']}")
    print(f"• Revision rounds: {profile['revision_rounds']}")
    print(f"• Late delivery penalty: {profile['late_penalty']}%")
    print(f"• Project difficulty: {profile['difficulty']}")
    print(f"• Region: {profile['region']}")
    print(f"• Client company size: {profile['company_size']}")
    
    confirm = input("\nIs this information correct? (y/n): ")
    if confirm.lower() != 'y':
        print("Let's try again.")
        return collect_user_responses()
    
    return profile

# Main application
def main():
    print("Welcome to the Digital Contract Advisor")
    print("This system will analyze your contract terms and provide guidance based on your situation.")
    
    # Load and process documents
    docs = load_contract_documents()
    process_documents(docs)
    
    # Build the graph
    graph = build_graph()
    
    # Collect user information
    user_profile = collect_user_responses()
    
    print("\nGenerating your personalized contract analysis. This may take a moment...")
    
    # Process the profile
    response = graph.invoke({"profile": user_profile})
    
    print("\n" + "="*50)
    print("Your Contract Analysis")
    print("="*50)
    print("\n" + response["analysis"])
    
    # Option to save the report
    save_option = input("\nWould you like to save this analysis to a file? (y/n): ")
    if save_option.lower() == 'y':
        # Create filename based on user profile details
        filename = f"contract_analysis_{user_profile['region'].replace(' ', '_').lower()}.pdf"
        create_pdf_report(user_profile, response["analysis"], filename)
    
    print("\nThank you for using the Digital Contract Advisor!")

if __name__ == "__main__":
    main()
