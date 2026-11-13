import fitz  # PyMuPDF
import openai
import os
import json
import logging
import traceback
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in the environment")

openai.api_key = OPENAI_API_KEY

class PentestReportParser:
    def __init__(self, logging_level=logging.INFO, timeout=30):
        """
        Initialize the parser with configurable logging and timeout
        
        :param logging_level: Logging level (default: INFO)
        :param timeout: Parsing timeout in seconds (default: 30)
        """
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout


    def get_vulnerabilty_data(self, pdf_path):
        """
        Process a PDF file and extract vulnerability information with safety checks
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        # Validate file exists and is not too large
        if not os.path.exists(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            return [], False
        
        file_size = os.path.getsize(pdf_path)
        if file_size > 75 * 1024 * 1024:  # 50MB limit
            self.logger.error(f"File too large: {file_size} bytes")
            return [], False

        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            clean_text = ' '.join(text.split())
            system_prompt = (
                "The following text contains various data entries extracted from a PDF regarding security vulnerabilities. "
                "Your task is to extract ALL entries completely, ensuring no information is truncated or summarized, even if entries are duplicated. "
                "Please organize all data into a valid JSON array, where each entry is a JSON object with the following keys: "
                "'Vulnerability name', 'CVSS Score', and 'Risk Score/Level'. Please do NOT omit any entries or fields. "
                "To ensure that all data from the input is completely captured, do NOT summarize even if information is repeated. "                
                "The output should be strictly JSON only, with no additional explanation, heading, or surrounding text. "
                "The output JSON should contain two keys : is_pentest_report and vulnerabilities."
                "The value for is_pentest_report should be set to True if the input pdf contains all of the keywords 'risk', 'remediation', 'penetration' and 'CVSS'"
                "The `vulnerabilities` should be a json list with keys following format with the respective information found in the uploaded document: [{'Vulnerability name': 'Example Vulnerability', 'CVSS Score': '7.8', 'Risk Score/Level': 'High'}]"
                "An example json response would look like:\n"
                """{
            "is_pentest_report": "true",
            "vulnerabilities": [{"Vulnerability name": "", "CVSS Score": '7.8', "Risk Score/Level": "High"}]
        }"""
            )

            response = openai.ChatCompletion.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": clean_text}
                ]
            )

            all_vulnerabilities = response['choices'][0]['message']['content']
            result_json = json.loads(all_vulnerabilities)
            # print(result_json)
            # with open('vulnerabilities_data.json', 'w') as file:
            #     json.dump(result_json, file, indent=4)
            return result_json['vulnerabilities'], result_json['is_pentest_report']

        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [], False
        
    def get_pentest_score(self, pdf_path):
        """
        Calculate pentest score with robust error handling
        """
        # Define risk mapping
        risk_mapping = {
            'Critical': 4,
            'High': 3,
            'Medium': 2,
            'Low': 1
        }

        try:
            vulnerability_data, is_pentest_report = self.get_vulnerabilty_data(pdf_path)
            # with open(pdf_path) as f:
            #     vulnerability_data = json.load(f)
            
            if not vulnerability_data:
                return 0.0, [], is_pentest_report

            vulnerability_score = 0.0
            total = 0
            filtered_data = []
            is_pentest_report = is_pentest_report or len(vulnerability_data) > 0
            for vulnerability in vulnerability_data:
                cvss_score = 0 if (vulnerability["CVSS Score"] == "" or vulnerability["CVSS Score"] == "NA" or vulnerability["CVSS Score"] == "N/A" or vulnerability["CVSS Score"] == "Not Available") else float(vulnerability["CVSS Score"])
                if cvss_score != 0:
                    risk_level = risk_mapping.get(vulnerability["Risk Score/Level"].split()[0], 0)
                    
                    vulnerability_score += cvss_score * risk_level
                    total = total + 1
                
                risk_level = vulnerability.get("Risk Score/Level", "").strip()
                cvss_score = vulnerability.get("CVSS Score", "").strip()

                try:
                    score = float(cvss_score)
                    if risk_level == "Critical" and score >= 8.0:
                        filtered_data.append(vulnerability)
                except ValueError:
                    # Skip if CVSS Score is "NA" or not a valid float
                    continue

            # Prevent division by zero            
            vulnerability_score = vulnerability_score / total if total > 0 else 0.0

            with open('./filtered_vulnerability_data.json', 'w') as f:
                json.dump(filtered_data, f, indent=4)

            return vulnerability_score, filtered_data, is_pentest_report
            # return vulnerability_score, vulnerability_data, 

        except Exception as e:
            self.logger.error(f"Score calculation error: {str(e)}")
            return 0.0, []

def main():
    parser = PentestReportParser()

    # Example usage for single PDF
    pdf_path = "Pen Test Vulnerabilities only.pdf"
    
    score, filtereed_data, vulnerabilty_data = parser.get_pentest_score(pdf_path)

    # if isinstance(vulnerabilty_data, str):
    #     result = json.loads(vulnerabilty_data)
    # else:
    #     result = vulnerabilty_data
    # with open('./filtered_vulnerability_data.json', 'w') as f:
    #     json.dump(result, f, indent=4)
    print(score)
    print()
    print()
    print()
    print(vulnerabilty_data)
    print()
    print()
    print()
    print(filtereed_data)
# def main():
#     parser = PentestReportParser()

#     pdf_list = [
#             "A-LIGN_SPoC_Penetration_Test.pdf",
#             "Bishop Fox Assessment Report - Winston Privacy.pdf",
#             "conti-cyber-attack-on-the-hse-full-report.pdf",
#             "CyHy Sample Report_508C.pdf",
#             "HackerOne Pentest Challenge Report - 2020-03-31.pdf",
#             "HackTheBox-Inlanefright-internal-penetration-test-report.pdf",
#             "LogicalTrust-OPNsense-security-assessment-report.pdf",
#             "long-penetration-test-report.pdf",
#             "opensips-audit-technical-report-full.pdf",
#             "Penetration-Test-Report-KeeWeb-by-Hackmanit.pdf",
#             "realvnc-penetration-test.pdf",
#             "SPoC_Penetration-Test.pdf",
#             "Veracode_-_GlobaLeaks_and_Tor2web.pdf",
#             "X41-Backstage-Audit-2024-Final-Report-Public.pdf",
#             "X41-Balrog-Review-2018-Final-Report-Public.pdf",
#             "Pen Test Vulnerabilities only.pdf"
#         ]

#     pdf_base_path = "/Users/belagaam/Desktop/Cyber Risk Backend/Penetration Test Reports/Done"

#     # Base JSON template (unchanged fields)
#     base_output = {
#         "businessQuestions": {
#         "name": "Shivam Kaushal",
#         "phone": "9929371862",
#         "email": "shivam.k@ekarigar.com",
#         "companyName": "ekarigarrr",
#         "country": "India",
#         "region": "Karnataka",
#         "primaryIndustry": "A",
#         "PrimeryIndusteryName": "(A) Agriculture, Forestry, Fishing",
#         "employeeCount": "51",
#         "annualRevenue": "20000",
#         "revenue2023": "222222",
#         "revenue2022": "222222",
#         "cybersecurityBudget": "",
#         "regulations": [
#             "None"
#         ],
#         "personalRecords": [
#             "Payment Card Information (PCI)"
#         ],
#         "recentBreach": "No",
#         "incidentCount": "",
#         "primaryCause": "",
#         "dataExposed": [],
#         "recordsExposed": "",
#         "incidentFixed": "",
#         "hasInsurance": "No",
#         "insuranceCarrier": "",
#         "insuranceCoverage": [
#             "Business Interruption Loss"
#         ],
#         "requestedCoverage": [],
#         "existingDeductibles": "",
#         "Payment Card Information (PCI)": {
#             "records": "22",
#             "recordsExposed": "0"
#         },
#         "Protected Health Information (PHI)": {
#             "records": "0",
#             "recordsExposed": "0"
#         },
#         "Personally Identifiable Information (PII)": {
#             "records": "0",
#             "recordsExposed": "0"
#         },
#         "Business Interruption Loss": {
#             "coverageAmount": "666",
#             "incomeProfitCover": "222",
#             "incomeProfitDeductible": "",
#             "contingentLossCover": "222",
#             "contingentLossDeductible": "",
#             "continuingExpenseCover": "222",
#             "continuingExpenseDeductible": ""
#         },
#         "Privacy Liability Coverage": {
#             "coverageAmount": "",
#             "deductibleAmount": "",
#             "regulatoryFinesCover": "",
#             "regulatoryFinesDeductible": "",
#             "regulatoryPenaltiesCover": "",
#             "regulatoryPenaltiesDeductible": "",
#             "legalExpensesCover": "",
#             "legalExpensesDeductible": ""
#         },
#         "Incident Response Costs": {
#             "coverageAmount": "",
#             "deductibleAmount": ""
#         },
#         "countryCode": "+91",
#         "cyberIncident": "No",
#         "numIncident": "",
#         "CCPA": "",
#         "BIPA": "",
#         "GDPR": "",
#         "UK_ICO": "",
#         "OPC_CA": "",
#         "NYDFS_Part_500": "",
#         "NYDFS_revenue": "",
#         "NYDFS_emp": ""
#     },
#         "assessmentData": {
#             "assessmentType": "PenTest",
#             "assessmentScore": None,          # will be replaced
#             "assessmentfileUrl": "https://cyber-insurance-bucket.s3.ap-south-1.amazonaws.com/assessments/PenTest_Vulnerabilities_only.pdf",          # optional: update if needed
#             "vulnerabilityData": []           # will be replaced
#         },
#         "securityQuestions": {
#         "pentestCriticalResolved": "",
#         "pentestCritical30Days": "",
#         "useMFA": [
#             "None"
#         ],
#         "remoteAccess": "No",
#         "remoteAccessTypes": [],
#         "securityTraining": "We do not offer any training",
#         "alternativeTraining": "",
#         "incidentResponseTraining": "",
#         "computerSafety": "",
#         "documentProcedure": "",
#         "fixProblems": "No",
#         "updatesIn30Days": "",
#         "dataEncryption": "No",
#         "dataBacking": "We do not backup our files",
#         "thirdPartyControls": "No",
#         "edrTools": "",
#         "edrToolsUsed": [
#             "None"
#         ],
#         "handlingAlerts": "",
#         "specialAccess": [],
#         "lockAccount": "No",
#         "noReceiveAlertApp": True,
#         "receiveAlert": "",
#         "incidentFile": "",
#         "noIncidentResponse": True
#     }
#     }

#     for pdf_file in pdf_list:
#         pdf_path = os.path.join(pdf_base_path, pdf_file)

#         print(f"Processing: {pdf_file}")

#         try:
#             score, vulnerability_data, is_pentest_report = parser.get_pentest_score(pdf_path)

#             # Create a deep copy so each PDF has independent data
#             output_data = json.loads(json.dumps(base_output))

#             # 🔁 Replace ONLY what you asked for
#             output_data["assessmentData"]["assessmentScore"] = score
#             output_data["assessmentData"]["vulnerabilityData"] = vulnerability_data
#             print(f"{pdf_path}: Vulnerabilities: {len(vulnerability_data)}, is_pentest_report: {is_pentest_report} ")

#             # Output filename: <pdf_name>.json
#             json_filename = f"{os.path.splitext(pdf_file)[0]}.json"
#             json_output_path = os.path.join("/Users/belagaam/Desktop/Cyber Risk Backend/cyber-insurance-backend/testing/reports/", json_filename)

#             with open(json_output_path, "w", encoding="utf-8") as f:
#                 json.dump(output_data, f, indent=2)

#             print(f"Saved → {json_filename}")

#         except Exception as e:
#             print(f"❌ Failed processing {pdf_file}: {str(e)}")

if __name__ == "__main__":
    main()

