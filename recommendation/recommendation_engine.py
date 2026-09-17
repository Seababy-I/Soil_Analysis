import json
from pathlib import Path


# =========================================================
# Load Agricultural Knowledge Base
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "agricultural_knowledge_base.json"

with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
    KNOWLEDGE_BASE = json.load(file)


# =========================================================
# Nutrient Classification
# =========================================================

def classify_nutrient(value, nutrient):
    """
    Classify nitrogen, phosphorus or potassium
    as Low, Medium or High.
    """

    config = KNOWLEDGE_BASE["nutrients"][nutrient]

    if value < config["low_max"]:
        return "Low"

    elif value <= config["medium_max"]:
        return "Medium"

    else:
        return "High"


# =========================================================
# pH Classification
# =========================================================

def classify_ph(ph):
    """
    Classify soil pH as Acidic, Suitable or Alkaline.
    """

    config = KNOWLEDGE_BASE["ph"]

    if ph < config["acidic_max"]:
        return "Acidic"

    elif ph <= config["suitable_max"]:
        return "Suitable"

    else:
        return "Alkaline"


# =========================================================
# Organic Carbon Classification
# =========================================================

def classify_organic_carbon(value):
    """
    Classify organic carbon as Low, Medium or High.
    """

    config = KNOWLEDGE_BASE["organic_carbon"]

    if value < config["low_max"]:
        return "Low"

    elif value <= config["medium_max"]:
        return "Medium"

    else:
        return "High"


# =========================================================
# Nutrient Recommendations
# =========================================================

def get_nutrient_recommendations(nutrient, status):
    """
    Generate recommendations based on nutrient status.
    """

    recommendations = []

    if status == "Low":

        # Use recommendations stored in the knowledge base
        recommendations.extend(
            KNOWLEDGE_BASE["nutrients"][nutrient]["recommendation"]
        )

    elif status == "Medium":

        recommendations.append(
            "Maintain balanced nutrient management"
        )

        recommendations.append(
            "Follow the crop-specific recommended fertilizer dose"
        )

    else:

        recommendations.append(
            "Avoid unnecessary additional fertilizer application"
        )

        recommendations.append(
            "Follow soil-test-based and crop-specific recommendations"
        )

    return recommendations


# =========================================================
# pH Recommendations
# =========================================================

def get_ph_recommendations(status):
    """
    Generate recommendations based on soil pH.
    """

    if status == "Acidic":

        return [
            "Consider the crop's pH tolerance",
            "Verify the soil-test result before amendment decisions",
            "Use locally recommended liming or soil-amendment practices",
            "Avoid applying a universal amendment dose"
        ]

    elif status == "Suitable":

        return [
            "No generic pH correction is indicated",
            "Maintain balanced soil and nutrient management"
        ]

    else:

        return [
            "Consider the crop's pH tolerance",
            "Verify the soil-test result before amendment decisions",
            "Use locally recommended soil-amendment practices",
            "Avoid applying a universal amendment dose"
        ]


# =========================================================
# Organic Carbon Recommendations
# =========================================================

def get_organic_carbon_recommendations(status):
    """
    Generate recommendations based on organic carbon status.
    """

    if status == "Low":

        return [
            "Increase organic matter inputs using suitable FYM or compost",
            "Consider crop-residue recycling where appropriate",
            "Consider green manuring where suitable",
            "Consider integrated nutrient management practices"
        ]

    elif status == "Medium":

        return [
            "Maintain organic matter through crop-residue recycling",
            "Use suitable organic inputs where appropriate"
        ]

    else:

        return [
            "Maintain current organic matter management practices"
        ]


# =========================================================
# Main Recommendation Engine
# =========================================================

def generate_recommendations(soil_data):
    """
    Generate soil classification and recommendations.

    Expected input:
        nitrogen       : kg/ha
        phosphorus     : kg/ha
        potassium      : kg/ha
        ph             : pH
        organic_carbon : percentage
    """

    required_parameters = [
        "nitrogen",
        "phosphorus",
        "potassium",
        "ph",
        "organic_carbon"
    ]

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    for parameter in required_parameters:

        if parameter not in soil_data:
            raise ValueError(
                f"Missing soil parameter: {parameter}"
            )

        if not isinstance(soil_data[parameter], (int, float)):
            raise TypeError(
                f"{parameter} must be a numeric value"
            )

    # -----------------------------------------------------
    # Classify soil parameters
    # -----------------------------------------------------

    nitrogen_status = classify_nutrient(
        soil_data["nitrogen"],
        "nitrogen"
    )

    phosphorus_status = classify_nutrient(
        soil_data["phosphorus"],
        "phosphorus"
    )

    potassium_status = classify_nutrient(
        soil_data["potassium"],
        "potassium"
    )

    ph_status = classify_ph(
        soil_data["ph"]
    )

    organic_carbon_status = classify_organic_carbon(
        soil_data["organic_carbon"]
    )

    # -----------------------------------------------------
    # Identify deficiencies
    # -----------------------------------------------------

    deficiencies = []

    if nitrogen_status == "Low":
        deficiencies.append("Nitrogen")

    if phosphorus_status == "Low":
        deficiencies.append("Phosphorus")

    if potassium_status == "Low":
        deficiencies.append("Potassium")

    if organic_carbon_status == "Low":
        deficiencies.append("Organic Carbon")

    # -----------------------------------------------------
    # Generate recommendations
    # -----------------------------------------------------

    recommendations = {

        "nitrogen": get_nutrient_recommendations(
            "nitrogen",
            nitrogen_status
        ),

        "phosphorus": get_nutrient_recommendations(
            "phosphorus",
            phosphorus_status
        ),

        "potassium": get_nutrient_recommendations(
            "potassium",
            potassium_status
        ),

        "ph": get_ph_recommendations(
            ph_status
        ),

        "organic_carbon": get_organic_carbon_recommendations(
            organic_carbon_status
        )
    }

    # -----------------------------------------------------
    # Fertilizer principles
    # -----------------------------------------------------

    fertilizer_principles = KNOWLEDGE_BASE[
        "fertilizer_principles"
    ]

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {

        "soil_status": {

            "nitrogen": nitrogen_status,

            "phosphorus": phosphorus_status,

            "potassium": potassium_status,

            "ph": ph_status,

            "organic_carbon": organic_carbon_status
        },

        "deficiencies": deficiencies,

        "recommendations": recommendations,

        "fertilizer_principles": fertilizer_principles
    }


# =========================================================
# Test the Recommendation Engine
# =========================================================

if __name__ == "__main__":

    # Sample soil input
    sample_soil = {

        "nitrogen": 200,

        "phosphorus": 8,

        "potassium": 100,

        "ph": 5.8,

        "organic_carbon": 0.4
    }

    # Generate recommendations
    result = generate_recommendations(sample_soil)

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\n========================================")
    print("       SOIL RECOMMENDATION ENGINE")
    print("========================================")

    print("\nSoil Status:")

    for parameter, status in result["soil_status"].items():
        print(f"  {parameter}: {status}")

    print("\nDetected Deficiencies:")

    if result["deficiencies"]:

        for deficiency in result["deficiencies"]:
            print(f"  - {deficiency}")

    else:

        print("  No major deficiencies detected.")

    print("\nRecommendations:")

    for parameter, recommendations in result["recommendations"].items():

        print(f"\n{parameter.upper()}:")

        for recommendation in recommendations:
            print(f"  - {recommendation}")

    print("\nFertilizer Framework:")

    print(
        f"  {result['fertilizer_principles']['framework']}"
    )

    print("\nFertilizer Principles:")

    for principle in result["fertilizer_principles"]["principles"]:
        print(f"  - {principle}")

    print("\n========================================")
    print("             TEST COMPLETE")
    print("========================================\n")