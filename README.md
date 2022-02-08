[Presentation](https://www.youtube.com/embed/0FFBhZnNI_M)

## Data

For those who are actively looking for data scientist jobs in the U.S., the best news this month is the LinkedIn Workforce Report August 2018. According to the report, there is a shortage of 151,717 people with data science skills, with particularly acute shortages in New York City, San Francisco Bay Area and Los Angeles. To help job hunters to better understand the job market, Shanshan Lu scraped Indeed website and collected information of 7,000 data scientist jobs around the U.S. on August 3rd. The [information](https://www.kaggle.com/sl6149/data-scientist-job-market-in-the-us) that he collected are: Company Name, Position Name, Location, Job Description, and Number of Reviews of the Company.

## Others' Work

I am on schedule to graduate with a PhD in EE this December and I've begun applying to machine learning jobs. Something that has confused me a bit is how many different titles exist for very similar jobs (i.e., data scientists and machine learning engineers will share many technical qualifications such as SQL, Python, Torch/Tensorflow, R).

Given Kaggle's active communitty, a number of coding repos are available that ask the following questions already, including [Exploration - Data Scientist job market
](https://www.kaggle.com/kambojharyana/exploration-data-scientist-job-market):

- Who gets hired? What kind of talent do employers want when they are hiring a data scientist?
- Which location has the most opportunities?
- What skills, tools, degrees or majors do employers want the most for data scientists?
- What's the difference between data scientist, data engineer and data analyst?
- Can you develop an efficient classification algorithm to differentiate the three job types above?

[Data Scientist Job Market(U.S)-Data Viz](https://www.kaggle.com/carriech/data-scientist-job-market-u-s-data-viz):

- What position names are the most common?
- What locations have the most jobs?
- What companies have the most jobs?

[MinJobQualification](https://www.kaggle.com/garyongguanjie/minjobqualification):

- What is the minimum education level required for different positions?

## My Questions

While the classifier repository achieved a ~77% classification precision using job descriptions to classify position name, they didn't perform an analysis on that relationship revealing what positions require what responsabilities/requirements/whatever else is included in a description. My research questions for this data vis project are:

- What position(s) are right for me, my skills, and my experience?
- How strongly are job description and position names correlated? In other words, can I achieve more than 77% classification precision?
- What types of positions are difficult to classify given descriptions?

Googling the most common job positions, here are their definitions:

- Data scientist: Data scientists work closely with business stakeholders to understand their goals and determine how data can be used to achieve those goals. They design data modeling processes, create algorithms and predictive models to extract the data the business needs, and help analyze the data and share insights with peers.
- Data analyst: The data analyst serves as a gatekeeper for an organization's data so stakeholders can understand data and use it to make strategic business decisions. It is a technical role that requires an undergraduate degree or master's degree in analytics, computer modeling, science, or math.
- Machine learning engineer: Machine learning engineers develop self-running AI software to automate predictive models for recommended searches, virtual assistants, translation apps, chatbots, and driverless cars. They design machine learning systems, apply algorithms to generate accurate predictions, and resolve data set problems.
- Data Science Manager: The data science manager is responsible for helping organization leverage on data, working with and through a team of data scientists and engineers to provide valuable direction and insight, for management to make informed decisions.

# Progress Reports

## 10/11/21-10/13/21

I've found out that linear SVM classifiers are indeed the best for NLP generally, however I found that a bagging ensemble of 30 classifiers improved classification precision from 77% to 95%. Additionally, I've found that incorrect classifications consistently occur most commonly with the "Data analyst" and "Manager" positions because they have the least amount of data, not because the job descriptions are poorly correlated to the job titles. I've inspected the job descriptions of incorrect classifications for these two classes and they seem fine, including highly correlated keywords such as "team work" and "leading" for the "manager" class. I'm going to begin work for the next work week on visualizing an interactive confusion matrix plot using "cm.csv" in D3 using react. Forked this project from Dr. Kelleher's [Vega-Lite API Template](https://vizhub.com/curran/717a939bb09b4b3297b62c20d42ea6a3), created this [Python repo for generating viz data](https://github.com/kwmcclintick/datavis_finalproject), and created this [gist](https://gist.github.com/kwmcclintick/10e608193750f65b5f84b2d3f7247bfd) from the Python repo.

## 10/13/21-10/20/21

Made a prototype scatter plot of raw position title TSNE embeddings which are color coordinated by the assigned position group class. Added interaction to show the raw position title of the scatter point that the mouse covers. Looking at some of the "other" data points, I've adjusted the class assignments to be a bit more accurate. For instance, any position title with the word "executive" is now a "data science manager" class, where before they were "other".

[Screenshot](https://github.com/kwmcclintick/datavis_finalproject/blob/main/old.PNG)

## 10/20/21-10/27/21

Changed code from vega-lite to D3 by forking [Interactive Color Legend](https://vizhub.com/curran/8b699c4000704216a709adfeb38f2411), which comes with the added feature of highlighting classes when mousing over them in the legend. Added a large marker for each of the 5 classes average location for quicker digestion, but they lack visibility. Removed axis and grid.

[Screenshot](https://github.com/kwmcclintick/datavis_finalproject/blob/main/new.PNG)

## 10/27/21-11/3/21
Changed class centroid text from mean location to median location. Changed class centroid text to have color matching class legend, and to become transparent when mousing over legend. Added white rectangle background to all class centroid texts for increased visibliity. Added description plot, such that there are now two embedding plots that represent the inputs and outputs of the machine learning classifier.

[Screenshot](https://github.com/kwmcclintick/datavis_finalproject/blob/main/current.PNG)

## 11/3/21-11/10/21
Began making the confusion matrix by creating two intersecting scaleBands in index.js, which are called in a \<div\> and \<rect\> in ConfusionGroup.js, following [this bar chart example](https://codedaily.io/tutorials/Fundamentals-of-Rendering-Data-as-an-SVG-Bar-Graph-with-D3-and-scaleBand). Reformatted the confusion matrix csv file to consist of a 'row', 'column', and 'value' column for better use in scaleBand. No visual progress to report.

## 11/10/21-11/17/21
Gave up on scalebands in favor of \<rect\> objects. Created first draft of confusion matrix. Viz is too big to fork, so I've continued making screenshots.

[Screenshot](https://github.com/kwmcclintick/datavis_finalproject/blob/main/current.PNG)

## 11/24/21-12/1/21
Corrected an error with the confusion matrice's values, and added interactivity to the confusion matrix, highlighting the linear SVM's classification performance depending on which class is moused over in the legend.

