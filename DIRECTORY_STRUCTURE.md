# Project Directory Structure

## Root Level
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `results.md` - Summary of evaluation results
- `SLM.pdf` - Generated research paper
- `B22CS096_B22CI017_B22CS093_proposal (1).pdf` - Project proposal

## `/logs/`
Training and evaluation log files:
- `training_llama_1b.log` - Llama-3.2-1B training logs
- `training_smollm.log` - SmolLM-360M SFT training logs
- `training_qwen.log` - Qwen2.5-0.5B training logs
- `training_qwen_1.5b.log` - Qwen2.5-1.5B training logs
- `training.log` - Phi-3-Medium training logs
- `training_2.log` - Additional training logs
- `training_dpo.log` - DPO training logs (v1)
- `training_dpo_smollm.log` - SmolLM-360M DPO training logs
- `training_dpo_v2.log` - DPO training logs (v2)
- `baseline_eval.log` - Base model evaluation logs
- `eval_qwen_1.5b.log` - Qwen2.5-1.5B evaluation logs
- `report_gen.log` - Report generation logs

## `/figures/`
Generated plots and visualizations:
- `training_curves.pdf/png` - Training loss, accuracy, LR, entropy curves
- `dpo_training.pdf/png` - DPO-specific training metrics
- `evaluation_results.pdf/png` - Detailed evaluation results charts
- `model_comparison.pdf/png` - Model performance comparison
- `size_vs_performance.pdf/png` - Model size vs performance scatter plot

## `/notebooks/`
Jupyter notebooks for experiments:
- `FMGA_course_project_gemma.ipynb` - Gemma-3-270M fine-tuning experiments
- `LFM2_finetune.ipynb` - LiquidAI LFM2-350M fine-tuning

## `/data/`
Evaluation results and output examples:
- `evaluation_results_gemma_ft.json` - Gemma fine-tuned evaluation results
- `evaluation_results_lfm2.json` - LFM2 evaluation results
- `report_outputs.json` - Sample model outputs for qualitative analysis

## `/paper/`
LaTeX paper and related files:
- `paper.tex` - IEEE format research paper source
- `generate_plots.py` - Python script to generate all figures
- `*.pdf` - Copies of figures for LaTeX compilation

## `/scripts/`
Utility scripts:
- `check_trl.py` - Check TRL library installation
- `check_trl_dir.py` - Check TRL directory structure

## `/src/`
Source code for training and evaluation (existing)

---

**Organization Benefits:**
- Clear separation of concerns
- Easy to find specific file types
- Figures ready for paper compilation
- Logs organized for analysis
- Data files grouped together
