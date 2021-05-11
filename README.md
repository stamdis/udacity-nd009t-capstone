# Software Engineer Salary

This repository includes my submission for the final capstone project, part of the [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) program by Udacity, co-created with Kaggle and Amazon Web Services. The program syllabus contains advanced machine learning techniques and algorithms applied to Amazon SageMaker.

The Nanodegree was awarded to me after being selected among the 325 top scorers based on performance in the scholarship's [AWS Machine Learning Engineer Scholarship](https://www.udacity.com/scholarships/aws-machine-learning-scholarship-program) Foundations Course and quiz exam.

The project was submitted on October 29, 2020.

## Introduction

During September 2020, a YouTube channel named [SocialNerds](https://www.youtube.com/channel/UCd5jW000te6bExqYth4TIxQ) released an anonymized [dataset](https://docs.google.com/spreadsheets/d/1TVL6IfF9yaEKa3S6ma69pn-6o2YFxzUgEMTdiec8BpU/edit?usp=sharing) of nearly 600 entries that describes salary levels of software engineers. The data was collected online during the summer of 2020 through a Google Forms questionnaire & commented upon on a [video](https://www.youtube.com/watch?v=e-83bz4RhQ4). The participants are Greek software engineers working mostly for companies located in Greece or abroad.

This project is an attempt to answer the following question:

> What should be the expected annual salary for a software technologist given specific criteria that describe both the employer and the employee?

The suggested solution includes the creation of a machine learning model based on current market data. The model should accept various information about a job candidate (software technologist) and a job description (company) while the output of the model should be an estimation about the expected salary level.

## Technologies

Amazon SageMaker, Python, Jupyter notebooks, NumPy, pandas.

## Content

The repository has a simple folder structure, each part is described in the sections below.

### Documents

In the [documents](documents) folder, you can find the project [proposal](documents/proposal.md) ([pdf](documents/proposal.pdf)) as well as the [report](documents/capstone.md) ([pdf](documents/capstone.pdf)) and [citations](documents/citations.md) of the final capstone project submission.

All available documents can be found in Markdown format. Counterpart documents in PDF format were generated using [Pandoc](https://pandoc.org/) with the following command:

```bash
pandoc --pdf-engine=xelatex proposal.md -o proposal.pdf
pandoc --pdf-engine=xelatex capstone.md -o capstone.pdf
```

### Notebooks

In the [notebooks](notebooks) folder, you can find the source code of the project in the form of Jupyter Notebooks as well as the dataset that was used.

The source code is splitted into multiple notebooks, each representing a distinct stage in the machine learning pipeline.

The original dataset was transformed according to each pipeline stage. Multiple versions of the dataset can be found, produced and consumed during each pipeline stage.

### Executed Notebooks

In the [output](output) folder, you can find the output of all notebooks after their execution in the Amazon Web Services environment. The notebooks were exported in HTML format.

### Resources

In the [resources](resources) folder, you can find the files that were used in the final [report](documents/capstone.md) ([pdf](documents/capstone.pdf)) of the capstone project.

## Disclaimer

The opinions expressed in this publication are those of the author(s). They do not purport to reflect the opinions or views of any aforementioned people or organization.
