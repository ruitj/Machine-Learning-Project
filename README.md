# NY WCB Injury Claim Prediction – Machine Learning Project
Repository for the **Machine Learning** course in the Master's program *Data Science and Advanced Analytics (2024/2025)* at **NOVA IMS**.

## Project Overview

This project aims to assist the **New York Workers’ Compensation Board (WCB)** in automating workplace injury claim assessments. The primary goal is to enhance **decision efficiency**.

The dataset includes over **590,000 claims**, featuring details on injury types, industries, wage info, and claim decisions. The task is to predict the most appropriate classification outcome based on available features.

## Features

**Accident Date** Injury date of the claim   
**Age at Injury Age** of injured worker when the injury occurred.  
**Alternative Dispute Resolution** Adjudication processes external to the Board.   
**Assembly Date** The date the claim was first assembled.   
**Attorney/ Representative** Is the claim being represented by an Attorney?   
**Average Weekly Wage** The wage used to calculate workers’ compensation, disability, or an Paid Leave wage replacement benefits.      
**Birth Year** The reported year of birth of the injured worker.   
**C-2 Date** Date of receipt of the Employer's Report of Work-Related; Injury/Illness or equivalent (formerly Form C-2).   
**C-3 Date** Date Form C-3 (Employee Claim Form) was received.   
**Carrier Name**Name of primary insurance provider responsible for providing workers’ compensation coverage to the injured worker’s employer.   
**Carrier Type** Type of primary insurance provider responsible for providing workers’ compensation coverage.   
**Claim Identifier** Unique identifier for each claim, assigned by WCB.   
**County of Injury** Name of the New York County where the injury occurred.   
**COVID-19 Indicator** Indication that the claim may be associated with COVID-19.   
**District Name** Name of the WCB district office that oversees claims for that region or area of the state.   
**First Hearing Date** Date the first hearing was held on a claim at a WCB hearing location. A blank date means the claim has not yet had a hearing held.    
**Gender** The reported gender of the injured worker.   
**IME-4 Count** Number of IME-4 forms received per claim. The IME-4 form is the “Independent Examiner's Report of Independent Medical Examination” form.   
**Industry Code** NAICS code and descriptions are available at: https://www.naics.com/search-naics-codes-by-industry/.   
**Industry Code Description** 2-digit NAICS industry code description used to classify businesses according to their economic activity.   
**Medical Fee Region** Approximate region where the injured worker would receive medical service.   
**OIICS Nature of Injury Description** The OIICS nature of injury codes & descriptions are available at https://www.bls.gov/iif/oiics_manual_2007.pdf.   
**WCIO Cause of Injury Code** The WCIO cause of injury codes & descriptions are at https://www.wcio.org/Active%20PNC/WCIO_Cause_Table.pdf   
**WCIO Cause of Injury Description** See description of field above.    
**WCIO Nature of Injury Code** The WCIO nature of injury are available at https://www.wcio.org/Active%20PNC/WCIO_Nature_Table.pdf   
**WCIO Nature of Injury Description**See description of field above.   
**WCIO Part Of Body Code** The WCIO part of body codes & descriptions are available athttps://www.wcio.org/Active%20PNC/WCIO_Part_Table.pdf   
**WCIO Part Of Body Description** See description of field above.   
**Zip Code** The reported ZIP code of the injured worker’s home address.   
**Agreement Reached** Binary variable: Yes if there is an agreement without the involvement of the WCB -> unknown at the start of a claim.   
**WCB Decision** Multiclass variable: Decision of the WCB relative to the claim: “Accident” means that claim refers to workplace accident, “Occupational Disease” means illness from the workplace. -> requires WCB deliberation so it is unknown at start of claim.    
**Claim Injury Type** Main target variable: Deliberation of the WCB relative to benefits awarded to the claim. Numbering indicates severity   

## Techniques & Methodology

We applied **supervised machine learning** with multiple models to predict **claim injury types**, including:

- Multi Layer Perceptron  
- Random Forest  
- XGBoost  


