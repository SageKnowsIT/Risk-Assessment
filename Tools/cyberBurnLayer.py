import numpy as np
import pandas as pd
import os

    """
    Module: cyberBurnLayer.py
    Author: Sage N Clements, CISSP
    Date: July 24, 2025
    Copyright (c) 2025 Sage N Clements

    Description: This Python-based tool is designed to help cyber insurance professionals quantitatively assess risk by modeling the expected loss within specific insurance layers ("burn layers") for individual accounts.       Unlike generic industry benchmarking tools, this model dynamically simulates losses based on each client's unique characteristics, enabling tailored pricing, underwriting, and retention strategies.
    """

    def my_function():
        # This part of the function was added by Sage N Clements on 07.24.2025.
        pass


def simulate_burn_layers(client_input):
    # Step 1: Estimate severity distribution parameters (lognormal)
    base_mu = 14.5  # base mean (log-scale), ~exp(14.5) = $2M
    base_sigma = 1.0  # standard deviation

    # Adjust for cyber controls
    if client_input.get("Data Segmentation", False):
        base_sigma -= 0.1
    if client_input.get("Incident Response Plan", False):
        base_mu -= 0.2

    # Step 2: Simulate loss severities
    simulated_losses = np.random.lognormal(
        mean=base_mu,
        sigma=base_sigma,
        size=client_input.get("Num Simulations", 10000)
    )

    # Step 3: Calculate burn per layer
    attachment_points = client_input["Attachment Points"]
    layer_widths = client_input["Layer Widths"]
    frequency = client_input.get("Estimated Frequency", 0.3)

    layer_data = []
    for i in range(len(layer_widths)):
        attach = attachment_points[i]
        limit = attach + layer_widths[i]
        layer_burn = np.clip(np.minimum(simulated_losses, limit) - attach, 0, layer_widths[i])
        expected_burn = frequency * np.mean(layer_burn)

        layer_data.append({
            "Layer": f"${attach:,.0f} - ${limit:,.0f}",
            "Attachment": attach,
            "Limit": limit,
            "Expected Burn ($)": expected_burn
        })

    return pd.DataFrame(layer_data)

def export_to_excel(df, filename="Client_Cyber_Burn_Layers.xlsx"):
    output_path = os.path.join("/mnt/data", filename)
    df.to_excel(output_path, index=False)
    return output_path

# Example usage
if __name__ == "__main__":
    client_input = {
        "Industry": "Healthcare",
        "Annual Revenue": 250_000_000,
        "Employee Count": 800,
        "Records at Risk": 1_500_000,
        "Estimated Frequency": 0.35,
        "MFA Implemented": True,
        "Data Segmentation": True,
        "Incident Response Plan": False,
        "Cybersecurity Rating": 72,
        "Attachment Points": [0, 1_000_000, 5_000_000, 10_000_000],
        "Layer Widths": [1_000_000, 4_000_000, 5_000_000],
        "Num Simulations": 10000,
    }

    df = simulate_burn_layers(client_input)
    output_file = export_to_excel(df)
    print(f"Excel file saved to: {output_file}")
