# Sample E. coli genomes for AMR predictor testing
# Source: BV-BRC, real clinical isolates

## sample_susceptible.fna
- BV-BRC Genome ID: 562.98097
- BV-BRC URL: https://www.bv-brc.org/view/Genome/562.98097
- Genome length: 1.94 Mbp
- Resistant to: None (fully susceptible)
- Susceptible to: ampicillin, cefotaxime, ceftazidime, cefuroxime, ciprofloxacin, gentamicin, piperacillin/tazobactam

## sample_mixed_resistance.fna
- BV-BRC Genome ID: 562.99403
- BV-BRC URL: https://www.bv-brc.org/view/Genome/562.99403
- Genome length: 1.78 Mbp
- Resistant to: ceftazidime, gentamicin
- Susceptible to: ampicillin, cefotaxime, cefuroxime, ciprofloxacin, piperacillin/tazobactam

## sample_multi_drug_resistant.fna
- BV-BRC Genome ID: 562.23792
- BV-BRC URL: https://www.bv-brc.org/view/Genome/562.23792
- Genome length: 2.02 Mbp
- Resistant to: ampicillin, amoxicillin/clavulanic acid, cefotaxime, ceftazidime, cefuroxime, ciprofloxacin, piperacillin/tazobactam
- Susceptible to: gentamicin


## Usage
Upload any of these `.fna` files to the Streamlit Predict page to test the model.
Compare model predictions with the known ground truth listed above.
