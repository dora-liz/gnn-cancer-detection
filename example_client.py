"""
Example client showing how to use the drug ranking API.

This demonstrates how a researcher would interact with the deployed model.
"""

import requests
import json


def get_drug_ranking(gene_expression: list, server_url: str = "http://localhost:5000"):
    """
    Send gene expression data to the API and get ranked drugs.
    
    Args:
        gene_expression: List of 19,215 gene expression values
        server_url: URL of the API server
        
    Returns:
        dict with ranked drugs
    """
    response = requests.post(
        f"{server_url}/predict",
        json={"gene_expression": gene_expression},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.json()}")


def main():
    # Example: Load a cell line's gene expression from a file
    # In real use, this would come from RNA-seq data
    
    import torch
    
    # Load an existing cell line as example (first cell line from training data)
    train_data = torch.load("train_data.pt", map_location="cpu", weights_only=False)
    example_genes = train_data["cell_line"].x[0].tolist()  # 19,215 values
    
    print("=" * 60)
    print("DRUG RANKING API CLIENT EXAMPLE")
    print("=" * 60)
    print(f"\nSending gene expression data ({len(example_genes)} features)...")
    
    try:
        result = get_drug_ranking(example_genes)
        
        print(f"\nResults for cell line:")
        print(f"  Total drugs scored: {result['num_drugs']}")
        print(f"\n  Top 10 Recommended Drugs:")
        print("  " + "-" * 50)
        
        for drug in result["ranked_drugs"][:10]:
            print(f"  #{drug['rank']:2d}: Drug_{drug['drug_index']:03d} | "
                  f"Predicted IC50: {drug['predicted_ic50']:.2f}")
        
        print("\n  " + "-" * 50)
        print(f"  Note: {result['note']}")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server!")
        print("   Make sure to run: python api_server.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
