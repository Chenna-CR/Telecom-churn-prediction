# Screenshot and Submission Checklist

Everything below needs **your own accounts or your own actions**. Nothing here has been done for you, and no screenshot, GitHub upload or LinkedIn activity has been created or faked.

## A. Fill in the editable placeholders (yellow highlight in the DOCX)
- [ ] Cover page: your name, registration number, programme and section, course code, faculty coordinator name with U.Id and designation, Lovely School name, project semester
- [ ] Certificate: student name, registration number, course code, supervisor signature, name, designation, school, date
- [ ] Declaration: your name, programme name, date, signature, registration number
- [ ] Acknowledgement: rewrite in your own words (the draft is only a starting point)
- [ ] Remove the yellow highlighting once filled in
- [ ] In Word: click inside the Table of Contents, press **F9** (Update entire table) and repeat for the List of Figures and List of Tables so that page numbers match your edits

## B. Genuine project screenshots (take them yourself after running the code)
- [ ] Terminal: `pip install -r requirements.txt` completing successfully
- [ ] Jupyter: notebook open with the objectives table at the top
- [ ] Jupyter: data quality cell output (missing values, blanks, duplicates)
- [ ] Jupyter: validation tuning output for the classifiers
- [ ] Jupyter: regression comparison table and classification comparison table
- [ ] Jupyter: any two figures (for example actual vs predicted and the priority map)
- [ ] Jupyter: final cell showing the saved model files
- [ ] File explorer: the project folder structure
- [ ] Terminal: `python src/predict.py ...` output (optional)

## C. GitHub repository (your account)
1. Create a new empty repository (suggested name: `telco-churn-predictive-analysis`). Do not add a README there; one is included.
2. In the project folder run:
   ```
   git init
   git add .
   git commit -m "Predictive analysis of telecom customer churn (CA2)"
   git branch -M main
   git remote add origin https://github.com/<your-username>/telco-churn-predictive-analysis.git
   git push -u origin main
   ```
3. Screenshots to take: [ ] repository home page showing the folder structure and README, [ ] the notebook rendered on GitHub, [ ] the commit history, [ ] the figures folder.
4. Put the repository URL on your report (for example in the Introduction) and in the viva presentation.
5. Check the dataset licence on the Kaggle/IBM page before making the repository public; if unsure, keep the data file out of the public repo (add `data/` to `.gitignore`) and keep the download link in the README.

## D. LinkedIn (your account, only if you want to share it)
- [ ] Post or project entry about the work, written by you, with the GitHub link
- [ ] Screenshot of the post or the Projects section
- [ ] Suggested honest framing: it is a learning project on a public benchmark dataset of a fictional telecom company

## E. Before submission
- [ ] Open the DOCX in Word, scroll through every page, check figures and tables are not split awkwardly
- [ ] Check total page count is between 30 and 80
- [ ] Read the whole report once and make sure you can explain every table and figure
- [ ] Rehearse the viva questions in `Viva_Preparation_Guide`
- [ ] Re-run the notebook from the first cell once on your own machine to confirm the results reproduce
