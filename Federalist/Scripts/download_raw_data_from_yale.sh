#!/bin/bash
# Federalist Papers Raw Data Download Script
#
# This script downloads the raw HTML content of all 85 Federalist Papers from
# the Yale Law School Avalon Project website. It downloads each paper as a
# separate file named with the format "N. Title - Author" (without .html extension).
#
# Usage:
#   cd Federalist
#   bash Scripts/download_raw_data_from_yale.sh
#
# Output:
#   Files are downloaded to the Raw/ directory (create this directory first)
#   Each file contains the HTML content of one Federalist Paper
#
# Note: This script should be run before prepare_federalist_data.py, as it
# provides the raw HTML files that the preparation script processes.

# Create Raw directory if it doesn't exist
mkdir -p Raw

# Download all Federalist Papers to Raw directory
wget -O "Raw/Raw/1. General Introduction - Hamilton" https://avalon.law.yale.edu/18th_century/fed01.asp
wget -O "Raw/Raw/2. Concerning Dangers from Foreign Force and Influence - Jay" https://avalon.law.yale.edu/18th_century/fed02.asp
wget -O "Raw/Raw/3. The Same Subject Continued: Concerning Dangers from Foreign Force and Influence - Jay" https://avalon.law.yale.edu/18th_century/fed03.asp
wget -O "Raw/Raw/4. The Same Subject Continued: Concerning Dangers from Foreign Force and Influence - Jay" https://avalon.law.yale.edu/18th_century/fed04.asp
wget -O "Raw/Raw/5. The Same Subject Continued: Concerning Dangers from Foreign Force and Influence - Jay" https://avalon.law.yale.edu/18th_century/fed05.asp
wget -O "Raw/Raw/6. Concerning Dangers from Dissensions Between the States - Hamilton" https://avalon.law.yale.edu/18th_century/fed06.asp
wget -O "Raw/Raw/7. The Same Subject Continued: Concerning Dangers from Dissensions Between the States - Hamilton" https://avalon.law.yale.edu/18th_century/fed07.asp
wget -O "Raw/Raw/8. The Consequences of Hostilities Between the States - Hamilton" https://avalon.law.yale.edu/18th_century/fed08.asp
wget -O "Raw/Raw/9. The Union as a Safeguard Against Domestic Faction and Insurrection - Hamilton" https://avalon.law.yale.edu/18th_century/fed09.asp
wget -O "Raw/Raw/10. The Same Subject Continued: The Union as a Safeguard Against Domestic Faction and Insurrection - Madison" https://avalon.law.yale.edu/18th_century/fed10.asp
wget -O "Raw/11. The Utility of the Union in Respect to Commercial Relations and a Navy - Hamilton" https://avalon.law.yale.edu/18th_century/fed11.asp
wget -O "Raw/12. The Utility of the Union in Respect to Revenue - Hamilton" https://avalon.law.yale.edu/18th_century/fed12.asp
wget -O "Raw/13. Advantage of the Union in Respect to Economy in Government - Hamilton" https://avalon.law.yale.edu/18th_century/fed13.asp
wget -O "Raw/14. Objections to the Proposed Constitution from Extent of Territory Answered - Madison" https://avalon.law.yale.edu/18th_century/fed14.asp
wget -O "Raw/15. The Insufficiency of the Present Confederation to Preserve the Union - Hamilton" https://avalon.law.yale.edu/18th_century/fed15.asp
wget -O "Raw/16. The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union - Hamilton" https://avalon.law.yale.edu/18th_century/fed16.asp
wget -O "Raw/17. The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union - Hamilton" https://avalon.law.yale.edu/18th_century/fed17.asp
wget -O "Raw/18. The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union - Hamilton and Madison" https://avalon.law.yale.edu/18th_century/fed18.asp
wget -O "Raw/19. The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union - Hamilton and Madison" https://avalon.law.yale.edu/18th_century/fed19.asp
wget -O "Raw/20. The Same Subject Continued: The Insufficiency of the Present Confederation to Preserve the Union - Hamilton and Madison" https://avalon.law.yale.edu/18th_century/fed20.asp
wget -O "Raw/21. Other Defects of the Present Confederation - Hamilton" https://avalon.law.yale.edu/18th_century/fed21.asp
wget -O "Raw/22. The Same Subject Continued: Other Defects of the Present Confederation - Hamilton" https://avalon.law.yale.edu/18th_century/fed22.asp
wget -O "Raw/23. The Necessity of a Government as Energetic as the One Proposed to the Preservation of the Union - Hamilton" https://avalon.law.yale.edu/18th_century/fed23.asp
wget -O "Raw/24. The Powers Necessary to the Common Defense Further Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed24.asp
wget -O "Raw/25. The Same Subject Continued: The Powers Necessary to the Common Defense Further Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed25.asp
wget -O "Raw/26. The Idea of Restraining the Legislative Authority in Regard to the Common Defense Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed26.asp
wget -O "Raw/27. The Same Subject Continued: The Idea of Restraining the Legislative Authority in Regard to the Common Defense Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed27.asp
wget -O "Raw/28. The Same Subject Continued: The Idea of Restraining the Legislative Authority in Regard to the Common Defense Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed28.asp
wget -O "Raw/29. Concerning the Militia - Hamilton" https://avalon.law.yale.edu/18th_century/fed29.asp
wget -O "Raw/30. Concerning the General Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed30.asp
wget -O "Raw/31. The Same Subject Continued: Concerning the Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed31.asp
wget -O "Raw/32. The Same Subject Continued: Concerning the Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed32.asp
wget -O "Raw/33. The Same Subject Continued: Concerning the Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed33.asp
wget -O "Raw/34. The Same Subject Continued: Concerning the Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed34.asp
wget -O "Raw/35. The Same Subject Continued: Concerning the Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed35.asp
wget -O "Raw/36. The Same Subject Continued: Concerning the Power of Taxation - Hamilton" https://avalon.law.yale.edu/18th_century/fed36.asp
wget -O "Raw/37. Concerning the Difficulties of the Convention in Devising a Proper Form of Government - Madison" https://avalon.law.yale.edu/18th_century/fed37.asp
wget -O "Raw/38. Incoherence of the Objections to the New Plan Exposed - Madison" https://avalon.law.yale.edu/18th_century/fed38.asp
wget -O "Raw/39. Conformity of the Plan to Republican Principles - Madison" https://avalon.law.yale.edu/18th_century/fed39.asp
wget -O "Raw/40. The Powers of the Convention to Form a Mixed Government Examined and Sustained - Madison" https://avalon.law.yale.edu/18th_century/fed40.asp
wget -O "Raw/41. General View of the Powers Conferred by the Constitution - Madison" https://avalon.law.yale.edu/18th_century/fed41.asp
wget -O "Raw/42. The Powers Conferred by the Constitution Further Considered - Madison" https://avalon.law.yale.edu/18th_century/fed42.asp
wget -O "Raw/43. The Same Subject Continued: The Powers Conferred by the Constitution Further Considered - Madison" https://avalon.law.yale.edu/18th_century/fed43.asp
wget -O "Raw/44. Restrictions on the Authority of the Several States - Madison" https://avalon.law.yale.edu/18th_century/fed44.asp
wget -O "Raw/45. The Alleged Danger From the Powers of the Union to the State Governments Considered - Madison" https://avalon.law.yale.edu/18th_century/fed45.asp
wget -O "Raw/46. The Influence of the State and Federal Governments Compared - Madison" https://avalon.law.yale.edu/18th_century/fed46.asp
wget -O "Raw/47. The Particular Structure of the New Government and Distribution of Power Among Its Different Parts - Madison" https://avalon.law.yale.edu/18th_century/fed47.asp
wget -O "Raw/48. These Departments Should Not Be So Far Separated as to Have No Constitutional Control Over Each Other - Madison" https://avalon.law.yale.edu/18th_century/fed48.asp
wget -O "Raw/49. Method of Guarding Against the Encroachments of Any One Department of Government by Appealing to the People Through a Convention - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed49.asp
wget -O "Raw/50. Periodic Appeals to the People Considered - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed50.asp
wget -O "Raw/51. The Structure of the Government Must Furnish the Proper Checks and Balances Between the Different Departments - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed51.asp
wget -O "Raw/52. The House of Representatives - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed52.asp
wget -O "Raw/53. The Same Subject Continued: The House of Representatives - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed53.asp
wget -O "Raw/54. The Apportionment of Members Among States - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed54.asp
wget -O "Raw/55. The Total Number of the House of Representatives - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed55.asp
wget -O "Raw/56. The Same Subject Continued: The Total Number of the House of Representatives - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed56.asp
wget -O "Raw/57. The Alleged Tendency of the Plan to Elevate the Few at the Expense of the Many Considered in Connection with Representation - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed57.asp
wget -O "Raw/58. Objection that the Number of Members Will Not Be Augmented as the Progress of Population Demands Considered - Madison" https://avalon.law.yale.edu/18th_century/fed58.asp
wget -O "Raw/59. Concerning the Power of Congress to Regulate the Election of Members - Hamilton" https://avalon.law.yale.edu/18th_century/fed59.asp
wget -O "Raw/60. The Same Subject Continued: Concerning the Power of Congress to Regulate the Election of Members - Hamilton" https://avalon.law.yale.edu/18th_century/fed60.asp
wget -O "Raw/61. The Same Subject Continued: Concerning the Power of Congress to Regulate the Election of Members - Hamilton" https://avalon.law.yale.edu/18th_century/fed61.asp
wget -O "Raw/62. The Senate - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed62.asp
wget -O "Raw/63. The Senate Continued - Hamilton or Madison" https://avalon.law.yale.edu/18th_century/fed63.asp
wget -O "Raw/64. The Powers of the Senate - Jay" https://avalon.law.yale.edu/18th_century/fed64.asp
wget -O "Raw/65. The Powers of the Senate Continued - Hamilton" https://avalon.law.yale.edu/18th_century/fed65.asp
wget -O "Raw/66. Objections to the Power of the Senate To Set as a Court for Impeachments Further Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed66.asp
wget -O "Raw/67. The Executive Department - Hamilton" https://avalon.law.yale.edu/18th_century/fed67.asp
wget -O "Raw/68. The Mode of Electing the President - Hamilton" https://avalon.law.yale.edu/18th_century/fed68.asp
wget -O "Raw/69. The Real Character of the Executive - Hamilton" https://avalon.law.yale.edu/18th_century/fed69.asp
wget -O "Raw/70. The Executive Department Further Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed70.asp
wget -O "Raw/71. The Duration in Office of the Executive - Hamilton" https://avalon.law.yale.edu/18th_century/fed71.asp
wget -O "Raw/72. The Same Subject Continued, and Re-Eligibility of the Executive Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed72.asp
wget -O "Raw/73. The Provision for Support of the Executive, and the Veto Power - Hamilton" https://avalon.law.yale.edu/18th_century/fed73.asp
wget -O "Raw/74. The Command of the Military and Naval Forces, and the Pardoning Power of the Executive - Hamilton" https://avalon.law.yale.edu/18th_century/fed74.asp
wget -O "Raw/75. The Treaty Making Power of the Executive - Hamilton" https://avalon.law.yale.edu/18th_century/fed75.asp
wget -O "Raw/76. The Appointing Power of the Executive - Hamilton" https://avalon.law.yale.edu/18th_century/fed76.asp
wget -O "Raw/77. The Appointing Power Continued and Other Powers of the Executive Considered - Hamilton" https://avalon.law.yale.edu/18th_century/fed77.asp
wget -O "Raw/78. The Judiciary Department - Hamilton" https://avalon.law.yale.edu/18th_century/fed78.asp
wget -O "Raw/79. The Judiciary Continued - Hamilton" https://avalon.law.yale.edu/18th_century/fed79.asp
wget -O "Raw/80. The Powers of the Judiciary - Hamilton" https://avalon.law.yale.edu/18th_century/fed80.asp
wget -O "Raw/81. The Judiciary Continued, and the Distribution of Judicial Authority - Hamilton" https://avalon.law.yale.edu/18th_century/fed81.asp
wget -O "Raw/82. The Judiciary Continued - Hamilton" https://avalon.law.yale.edu/18th_century/fed82.asp
wget -O "Raw/83. The Judiciary Continued in Relation to Trial by Jury - Hamilton" https://avalon.law.yale.edu/18th_century/fed83.asp
wget -O "Raw/84. Certain General and Miscellaneous Objections to the Constitution Considered and Answered - Hamilton" https://avalon.law.yale.edu/18th_century/fed84.asp
wget -O "Raw/85. Concluding Remarks - Hamilton" https://avalon.law.yale.edu/18th_century/fed85.asp
