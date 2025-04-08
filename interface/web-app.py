import streamlit as st
import torch
import pickle
import pandas as pd
import torch.nn.functional as F
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
import urllib.request

# Load the saved data
@st.cache_resource
def load_data():
    url = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/drug_embedding.pt"
    local_path = "drug_embedding.pt"
    urllib.request.urlretrieve(url, local_path)
    drug_embedding = torch.load(local_path, weights_only=True)

    url_disease = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/disease_embedding.pt"
    local_path_disease = "disease_embedding.pt"
    urllib.request.urlretrieve(url_disease, local_path_disease)
    disease_embedding = torch.load(local_path_disease, weights_only=True)

    # Load the other data files
    url_drug_nodes = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/drug_nodes.pkl"
    local_path_drug_nodes = "drug_nodes.pkl"
    urllib.request.urlretrieve(url_drug_nodes, local_path_drug_nodes)
    with open(local_path_drug_nodes, "rb") as f:
        drug_nodes = pickle.load(f)
    
    url_disease_nodes = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/disease_nodes.pkl"
    local_path_disease_nodes = "disease_nodes.pkl"
    urllib.request.urlretrieve(url_disease_nodes, local_path_disease_nodes)
    with open(local_path_disease_nodes, "rb") as f:
        disease_nodes = pickle.load(f)

    url_drug_embeddings = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/drug_embeddings.pkl"
    local_path_drug_embeddings = "drug_embeddings.pkl"
    urllib.request.urlretrieve(url_drug_embeddings, local_path_drug_embeddings)
    with open(local_path_drug_embeddings, "rb") as f:
        drug_embeddings = pickle.load(f)

    url_disease_embeddings = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/disease_embeddings.pkl"
    local_path_disease_embeddings = "disease_embeddings.pkl"
    urllib.request.urlretrieve(url_disease_embeddings, local_path_disease_embeddings)
    with open(local_path_disease_embeddings, "rb") as f:
        disease_embeddings = pickle.load(f)

    url_gene_embeddings = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/gene_embeddings.pkl"
    local_path_gene_embeddings = "gene_embeddings.pkl"
    urllib.request.urlretrieve(url_gene_embeddings, local_path_gene_embeddings)
    with open(local_path_gene_embeddings, "rb") as f:
        gene_embeddings = pickle.load(f)

    url_triples_df = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/triples_df.pkl"
    local_path_triples_df = "triples_df.pkl"
    urllib.request.urlretrieve(url_triples_df, local_path_triples_df)
    with open(local_path_triples_df, "rb") as f:
        triples_df = pickle.load(f)

    url_drug_disease_df = "https://raw.githubusercontent.com/Morgan11-tech/Machine-Learning-for-Precision-Medicine-in-Ghana/main/interface/drug_disease_df.csv"
    local_path_drug_disease_df = "drug_disease_df.csv"
    urllib.request.urlretrieve(url_drug_disease_df, local_path_drug_disease_df)
    
    drug_disease_df = pd.read_csv(local_path_drug_disease_df)
    
    

    return drug_embedding, disease_embedding, drug_nodes, disease_nodes, drug_disease_df, drug_embeddings, disease_embeddings, gene_embeddings, triples_df

# Load data
drug_embedding, disease_embedding, drug_nodes, disease_nodes, drug_disease_df, drug_embeddings, disease_embeddings, gene_embeddings, triples_df = load_data()

def separate_entities(triples_df):
    drugs = set()
    diseases = set()
    genes_of_interest = {'DPYD', 'UGT1A1', 'PIK3CA', 'NRAS', 'KRAS', 'BRAF'}
    genes = set()

    for entity in triples_df['drugs and diseases']:
        if triples_df[triples_df['drugs and diseases'] == entity]['relation_type'].str.contains('drug').any():
            drugs.add(entity)
        else:
            diseases.add(entity)
    
    for gene in triples_df['protein']:
        if gene in genes_of_interest:
            genes.add(gene)
    
    return list(drugs), list(diseases), list(genes)

def get_normalized_embeddings(entity_list, model):
    embeddings = {}
    for entity in entity_list:
        if entity in model.wv:
            embedding = torch.tensor(model.wv[entity])
            normalized_embedding = F.normalize(embedding, p=2, dim=0)
            embeddings[entity] = normalized_embedding.numpy()
    return embeddings

def build_known_associations(triples_df, genes_of_interest):
    drug_gene_associations = set()
    disease_gene_associations = set()
    
    for _, row in triples_df.iterrows():
        entity = row['drugs and diseases']
        gene = row['protein']
        relation = row['relation_type']
        
        if gene in genes_of_interest:
            if 'drug' in relation:
                drug_gene_associations.add((entity, gene))
            elif 'disease' in relation:
                disease_gene_associations.add((entity, gene))
    
    return drug_gene_associations, disease_gene_associations

def filter_known_associations(drug_name, potential_diseases, drug_gene_associations, disease_gene_associations, genes_of_interest):
    novel_associations = []
    drug_genes = set(gene for drug, gene in drug_gene_associations if drug == drug_name)
    
    for disease, score in potential_diseases:
        disease_genes = set(gene for dis, gene in disease_gene_associations if dis == disease)
        
        # If there's no overlap in genes and the disease is not a gene of interest, consider it a novel association
        if not drug_genes.intersection(disease_genes) and disease not in genes_of_interest:
            novel_associations.append((disease, score))
    
    return novel_associations



def get_similarity(disease, disease_nodes, disease_embedding):
    try:
        index = list(disease_nodes).index(disease)
        return disease_embedding[index]
    except ValueError:
        st.warning(f"Disease '{disease}' not found in disease nodes.")
        return torch.zeros(disease_embedding.shape[1])

def get_predicted_relation(disease_embedding, drug_indications, drug_contraindications, drug_off_label_uses, disease_nodes, target_disease):
    target_embedding = get_similarity(target_disease, disease_nodes, disease_embedding)
    
    indications_similarities = torch.tensor([F.cosine_similarity(target_embedding, get_similarity(indication, disease_nodes, disease_embedding), dim=0) for indication in drug_indications]) if drug_indications else torch.tensor([])
    contraindications_similarities = torch.tensor([F.cosine_similarity(target_embedding, get_similarity(contraindication, disease_nodes, disease_embedding), dim=0) for contraindication in drug_contraindications]) if drug_contraindications else torch.tensor([])
    off_label_similarities = torch.tensor([F.cosine_similarity(target_embedding, get_similarity(off_label, disease_nodes, disease_embedding), dim=0) for off_label in drug_off_label_uses]) if drug_off_label_uses else torch.tensor([])

    max_indication_sim = indications_similarities.max() if len(indications_similarities) > 0 else torch.tensor(-1.0)
    max_contraindication_sim = contraindications_similarities.max() if len(contraindications_similarities) > 0 else torch.tensor(-1.0)
    max_off_label_sim = off_label_similarities.max() if len(off_label_similarities) > 0 else torch.tensor(-1.0)

    max_sim = torch.max(torch.stack([max_indication_sim, max_contraindication_sim, max_off_label_sim]))

    if max_sim == max_indication_sim and len(indications_similarities) > 0:
        return 'Indication'
    elif max_sim == max_contraindication_sim and len(contraindications_similarities) > 0:
        return 'Contraindication'
    elif max_sim == max_off_label_sim and len(off_label_similarities) > 0:
        return 'Off-label use'
    else:
        return 'Unknown'

def find_novel_associations(drug_name, drug_embedding, disease_embedding, drug_indications, drug_contraindications, drug_off_label_uses, drug_nodes, disease_nodes, top_k=10):
    known_diseases = set(drug_indications + drug_contraindications + drug_off_label_uses)
    
    try:
        drug_index = list(drug_nodes).index(drug_name)
        drug_vec = drug_embedding[drug_index].unsqueeze(0)
    except ValueError:
        st.error(f"Drug '{drug_name}' not found in drug nodes.")
        return []

    similarities = F.cosine_similarity(drug_vec, disease_embedding);
    
    # Filter out known diseases and sort
    novel_similarities = [(disease, sim.item()) for disease, sim in zip(disease_nodes, similarities) if disease not in known_diseases]
    sorted_similarities = sorted(novel_similarities, key=lambda x: x[1], reverse=True)
    
    potential_associations = []
    for disease_id, score in sorted_similarities[:top_k]:
        predicted_relation = get_predicted_relation(disease_embedding, drug_indications, drug_contraindications, drug_off_label_uses, disease_nodes, disease_id)
        potential_associations.append((disease_id, score, predicted_relation))
    return potential_associations

def generate_pdf_report(drug_name, novel_associations):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Potential Novel Associations for Drug: {drug_name}", styles['Title']))
    elements.append(Paragraph("This report contains the potential novel associations of the selected drug based on the analysis.", styles['Normal']))
    
    data = [["Index", "Disease ID", "Score", "Predicted Relation"]]
    for i, (disease_id, score, predicted_relation) in enumerate(novel_associations, start=1):
        data.append([i, disease_id, f"{score:.2f}", predicted_relation])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    return buffer

def predict_for_drug(drug_name, drug_embeddings, disease_embeddings, gene_embeddings, 
                     drug_gene_associations, disease_gene_associations, genes_of_interest, top_n=10):
    if drug_name not in drug_embeddings:
        return None, None

    drug_embedding = drug_embeddings[drug_name]

    # Predict potential diseases
    drug_embedding_tensor = torch.tensor(drug_embedding).unsqueeze(0) 
    disease_similarities = torch.cosine_similarity(drug_embedding_tensor, torch.tensor(list(disease_embeddings.values())))
    disease_similarities = disease_similarities.squeeze()
    disease_scores = list(zip(disease_embeddings.keys(), disease_similarities))
    disease_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out known associations and genes of interest
    novel_diseases = filter_known_associations(drug_name, disease_scores, 
                                               drug_gene_associations, disease_gene_associations, genes_of_interest)
    potential_diseases = novel_diseases[:top_n]

    # Predict potential shared genes for each disease
    potential_genes = []
    for disease, _ in potential_diseases:
        disease_embedding = disease_embeddings[disease]
        disease_embedding_tensor = torch.tensor(disease_embedding).unsqueeze(0)
        gene_similarities = torch.cosine_similarity(disease_embedding_tensor, torch.tensor(list(gene_embeddings.values())))
        gene_similarities = gene_similarities.squeeze()
        gene_scores = list(zip(gene_embeddings.keys(), gene_similarities))
        gene_scores.sort(key=lambda x: x[1], reverse=True)
        potential_genes.append(gene_scores[0])  # Append the top gene for each disease

    return potential_diseases, potential_genes

# Prepare data for predictions
@st.cache_resource
def prepare_prediction_data():
    drugs, diseases, genes = separate_entities(triples_df)
    genes_of_interest = {'DPYD', 'UGT1A1', 'PIK3CA', 'NRAS', 'KRAS', 'BRAF'}
    
    drug_gene_associations, disease_gene_associations = build_known_associations(triples_df, genes_of_interest)
    
    return drugs, diseases, genes, genes_of_interest, drug_gene_associations, disease_gene_associations


# Generate PDF report for drug-disease-gene prediction
def generate_drug_disease_gene_pdf(drug_name, potential_diseases, potential_genes):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Drug-Disease-Gene Predictions for Drug: {drug_name}", styles['Title']))
    elements.append(Paragraph("This report contains the potential disease associations and shared genes for the selected drug based on the analysis.", styles['Normal']))
    
    data = [["Index", "Disease", "Disease Score", "Predicted Shared Gene"]]
    for i, ((disease, disease_score), (gene, _)) in enumerate(zip(potential_diseases, potential_genes), start=1):
        data.append([i, disease, f"{disease_score:.2f}", gene])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    return buffer

# Streamlit app
st.title("Drug-Disease Association Prediction")

st.write("This application predicts potential novel associations between drugs and diseases based on embeddings and known associations using node2vec and cosine similarity.")

# Create tabs
tab1, tab2 = st.tabs(["Drug-Disease Association", "Drug-Disease-Gene Prediction"])

# Drug-Disease Association tab
with tab1:
    # Drug selection
    drug_name = st.selectbox("Select a drug", list(drug_nodes))

    if drug_name:
        drug_triples = drug_disease_df[drug_disease_df['x_name'] == drug_name]

        # Get the indications and contraindications
        indications = drug_triples[drug_triples['relation'] == 'indication']['y_name'].tolist()
        contraindications = drug_triples[drug_triples['relation'] == 'contraindication']['y_name'].tolist()
        drug_off_label_uses = drug_triples[drug_triples['relation'] == 'off-label use']['y_name'].tolist()

        st.write(f"Number of known indications: {len(indications)}")
        st.write(f"Number of known contraindications: {len(contraindications)}")
        st.write(f"Number of known off-label uses: {len(drug_off_label_uses)}")

        # Find novel associations
        novel_associations = find_novel_associations(
            drug_name, 
            drug_embedding,
            disease_embedding, 
            indications, 
            contraindications, 
            drug_off_label_uses,
            drug_nodes,
            disease_nodes
        )

        st.write(f"Drug: {drug_name}")
        st.subheader("Potential Novel Associations:")
        # Create a DataFrame from the novel_associations list
        df = pd.DataFrame(
            [(i, disease_id, f'{score:.2f}', predicted_relation) 
            for i, (disease_id ,score, predicted_relation) in enumerate(novel_associations, start=1)],
            columns=["No.", "Disease", "Predicted Score", "Predicted Relation"]
        )

        # Set the "No." column as the index
        df.set_index("No.", inplace=True)

        # Display the table in Streamlit
        st.table(df)

        st.error("NB: These are potential associations and should be validated by domain experts.")

        # Generate PDF report
        pdf_report = generate_pdf_report(drug_name, novel_associations)

        # Add download button for PDF
        st.download_button(
            label="Download Associations Report as PDF",
            data=pdf_report,
            file_name=f"{drug_name}_novel_associations.pdf",
            mime="application/pdf"
        )

# Drug-Disease-Gene Prediction tab
with tab2:
    st.header("Drug-Disease-Gene Prediction")
    
    # Drug selection for gene prediction
    drug_name_gene = st.selectbox("Select a drug", list(drug_embeddings.keys()), key="drug_gene")

    if drug_name_gene:
        if st.button("Predict"):
            potential_diseases, potential_genes = predict_for_drug(drug_name_gene, drug_embeddings, disease_embeddings, gene_embeddings, 
                                                                   drug_gene_associations, disease_gene_associations, genes_of_interest)
            
            if potential_diseases is None or potential_genes is None:
                st.write(f"Drug '{drug_name_gene}' not found in the dataset.")
            else:
                st.write(f"Top 10 Potential Novel Associations for {drug_name_gene}:")
                
                # Create a DataFrame for display
                results = []
                for i, ((disease, disease_score), (gene, gene_score)) in enumerate(zip(potential_diseases, potential_genes), 1):
                    results.append({
                        'Disease': disease,
                        'Disease Score': f'{disease_score:.2f}',
                        'Predicted Shared Gene': gene
                    })
                
                results_df = pd.DataFrame(results)
                st.table(results_df)

                st.error("NB: These are potential novel associations and should be validated by domain experts.")

                # Generate PDF report for drug-disease-gene prediction
                pdf_report = generate_drug_disease_gene_pdf(drug_name_gene, potential_diseases, potential_genes)

                # Add download button for PDF
                st.download_button(
                    label="Download Drug-Disease-Gene Predictions Report as PDF",
                    data=pdf_report,
                    file_name=f"{drug_name_gene}_drug_disease_gene_predictions.pdf",
                    mime="application/pdf"
                )

