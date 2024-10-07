import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial import distance
from sklearn.cluster import KMeans
import numpy as np

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'].mean(dim=1).numpy()

def optimal_clusters(embeddings, max_clusters=10):
    wcss = []  # within-cluster sum of squares
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)

    # Compute the differences between successive values of WCSS
    differences = np.diff(wcss)
    
    # If all differences are small, clustering may not be beneficial.
    if all(diff > -0.1 for diff in differences):
        return 1
    
    # Return the cluster count at the "elbow" point, which is when differences start becoming smaller
    optimal_clusters = next((i for i, diff in enumerate(differences, 1) if diff > -0.1), 1)
    return optimal_clusters + 1

def is_anomaly(invoice_line, previous_invoice_lines, threshold=0.5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Compute embeddings for previous invoice lines
    embeddings = [get_embedding(line, model, tokenizer) for line in previous_invoice_lines]
   
    print(f"Embeddings: {embeddings.len()}")
 
    # Determine optimal clusters
    n_clusters = optimal_clusters(embeddings)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    # Compute embedding for the input invoice_line
    invoice_embedding = get_embedding(invoice_line, model, tokenizer)

    # Check if the distance to all centroids is greater than the threshold
    min_distance = min([distance.euclidean(centroid, invoice_embedding) for centroid in centroids])

    return min_distance > threshold

# Test
previous_invoice_lines = [
    "Purchase of widget A", "Transaction of widget A", "Acquisition of widget A",
    "Purchase of gadget X", "Transaction of gadget X", "Acquisition of gadget X"
]
invoice_line = "Free trip to Hawaii"
result = is_anomaly(invoice_line, previous_invoice_lines)
print(result)

