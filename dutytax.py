import os
import json
import httpx
import asyncio

from typing import Union, Optional

LEVY_UNITS = {
    "weight_in_": "weight",
    "volume_in_": "volume",
    "alcohol_by_volume": "alcohol content",
    "area_in_": "area",
    "quantity": "quantity",
    "length_in_": "length",
    "item_composition.steel": "steel content",
    "item_composition.aluminum": "aluminum content",
    "item_composition.copper": "copper content",
}


class Environment:
    def __init__(self, env: str):
        self.env = env
        if self.env == "prod":
            self.url = os.getenv("LANDED_COST_API_URL", "NO-URL-PROVIDED")
            self.key = os.getenv("LANDED_COST_API_KEY", "NO-KEY-PROVIDED")
        elif self.env == "dev":
            self.url = os.getenv("LANDED_COST_DEV_API_URL", "NO-URL-PROVIDED")
            self.key = os.getenv("LANDED_COST_DEV_API_KEY", "NO-KEY-PROVIDED")
        else:
            raise ValueError(f"Invalid environment: {self.env}")

        if self.url == "NO-URL-PROVIDED" or self.key == "NO-KEY-PROVIDED":
            raise ValueError(
                f"Environment is missing either a URL or credential key, double check configuration. url: {self.url}, key: {self.key if self.key == 'NO-KEY-PROVIDED' else self.key[0:5] + '...' + self.key[-2:]}"
            )

    def __str__(self):
        return self.env

    def __repr__(self):
        return self.env


class Item:
    def __init__(
        self,
        product_id: str,
        country_of_origin: str,
        hs_code: str,
        is_auto_part: bool = True,
        is_usmca: bool = False,
        contains_steel_copper_or_aluminum: bool = True,
    ):
        # Inputs
        self.product_id = product_id
        self.country_of_origin = country_of_origin.strip().upper()
        self.hs_code = hs_code.strip().upper()
        self.is_auto_part = is_auto_part
        self.is_usmca = (
            is_usmca if country_of_origin in ("US", "CA", "MX", "PR") else False
        )
        self.contains_steel_copper_or_aluminum = contains_steel_copper_or_aluminum

        # Standard Settings
        self.item_amount = 100000
        self.currency_code = "USD"
        self.quantity = 1
        self.sku = "TEST_ITEM"
        self.weight_unit = "KILOGRAM"
        self.weight_value = 10
        self.volume_unit = "LITER"
        self.volume_value = 10
        self.alcohol_by_volume_unit = "PERCENTAGE"
        self.alcohol_by_volume_value = 10

        # Set by results
        self.item_id = None
        self.duty_rate = None
        self.applied_levies = None
        self.referenced_units_of_measure = None
        self.tax_rate = None
        self.applied_taxes = None

    def to_graphql_item(self) -> dict:
        attributes = []
        if not self.is_auto_part:
            attributes.append({"key": "duty_exclusions", "value": "non_auto_parts"})
        if self.is_usmca:
            attributes.append({"key": "free_trade_agreements", "value": "usmca"})

        if self.contains_steel_copper_or_aluminum:
            composition = 0.33
        else:
            composition = 0.0

        product_composition = [
            {"material": "steel", "percentage": composition},
            {"material": "aluminum", "percentage": composition},
            {"material": "copper", "percentage": composition},
        ]

        self.item = {
            "productId": self.product_id,
            "amount": self.item_amount,
            "countryOfOrigin": self.country_of_origin,
            "currencyCode": self.currency_code,
            "hsCode": self.hs_code,
            "quantity": self.quantity,
            "sku": self.sku,
            "measurements": [
                {
                    "unitOfMeasure": self.weight_unit,
                    "value": self.weight_value,
                    "type": "WEIGHT",
                },
                {
                    "unitOfMeasure": self.volume_unit,
                    "value": self.volume_value,
                    "type": "VOLUME",
                },
                {
                    "unitOfMeasure": self.alcohol_by_volume_unit,
                    "value": self.alcohol_by_volume_value,
                    "type": "ALCOHOL_BY_VOLUME",
                },
            ],
            "productComposition": product_composition,
        }

        if len(attributes) > 0:
            self.item["attributes"] = attributes

        return self.item


class LandedCost:
    def __init__(
        self,
        ship_to_country: str,
        ship_from_country: str,
        ship_to_postal_code: Optional[str] = None,
        is_postal: bool = False,
        items: list[Item] = [],
    ):
        # Inputs
        self.ship_to_country = ship_to_country.strip().upper()
        self.ship_from_country = ship_from_country.strip().upper()
        self.ship_to_postal_code = ship_to_postal_code
        self.is_postal = is_postal
        self.items = items

        # Standard Settings
        self.shipping_amount = 100000
        self.currency_code = "USD"
        self.service_level_code = (
            "ups_standard" if not self.is_postal else "postal_deminimis_testing"
        )
        self.tariff_rate_type = "ZONOS_PREFERRED"
        self.end_use = "NOT_FOR_RESALE"
        self.calculation_method = "DDP_PREFERRED"

        self.graphql_query = self._get_graphql_query()
        self.graphql_variables = self._build_graphql_variables()

    def _get_graphql_query(self):
        with open("request.graphql", "r") as f:
            return f.read()

    def _build_graphql_variables(self) -> dict:
        # Party Create Workflow Inputs
        location = {
            "countryCode": self.ship_to_country,
        }

        if self.ship_to_country in ("CA", "BR"):
            location["administrativeAreaCode"] = (
                "QC" if self.ship_to_country == "CA" else "CE"
            )

        elif self.ship_to_country == "US":
            if self.ship_to_postal_code:
                location["postalCode"] = self.ship_to_postal_code

        destination_party = {
            "type": "DESTINATION",
            "location": location,
        }

        origin_party = {
            "type": "ORIGIN",
            "location": {"countryCode": self.ship_from_country},
        }

        partyCreateWorkflowInputs = [destination_party, origin_party]

        # Item Create Workflow Inputs
        itemCreateWorkflowInputs = [item.to_graphql_item() for item in self.items]

        # Shipment Rating Create Workflow Input
        shipmentRatingCreateWorkflowInput = {
            "amount": self.shipping_amount,
            "currencyCode": self.currency_code,
            "serviceLevelCode": self.service_level_code,
        }

        # Landed Cost Calculate Workflow Input
        landedCostCalculateWorkflowInput = {
            "currencyCode": self.currency_code,
            "tariffRate": self.tariff_rate_type,
            "endUse": self.end_use,
            "calculationMethod": self.calculation_method,
        }

        return {
            "partyCreateWorkflowInputs": partyCreateWorkflowInputs,
            "itemCreateWorkflowInputs": itemCreateWorkflowInputs,
            "shipmentRatingCreateWorkflowInput": shipmentRatingCreateWorkflowInput,
            "landedCostCalculateWorkflowInput": landedCostCalculateWorkflowInput,
        }

    async def _request(self, env: Union[Environment, str]) -> dict:
        if isinstance(env, str):
            env = Environment(env)

        async with httpx.AsyncClient(timeout=30.0) as client:
            body = json.dumps(
                {
                    "query": self.graphql_query,
                    "variables": self.graphql_variables,
                }
            )

            r = await client.post(
                env.url,
                data=body,
                headers={
                    "credentialToken": env.key,
                    "Content-Type": "application/json",
                },
            )
        r.raise_for_status()
        response = r.json()

        if "errors" in response or "code" in response:
            print(f"Error in response: {response}")

        return response

    async def process_items(self, env: Union[Environment, str]) -> list[Item]:
        if isinstance(env, str):
            env = Environment(env)

        response = await self._request(env)

        if "errors" in response.keys():
            raise Exception(response["errors"])

        data = response["data"]

        for item in self.items:
            # Get item id by matching productId
            for item_data in data["itemCreateWorkflow"]:
                if item_data["productId"] == item.product_id:
                    item_id = item_data["id"]
                    break

            # Find the duties and taxes for this item
            total_duties = 0
            total_taxes = 0
            applied_levies = []
            applied_taxes = []
            referenced_units_of_measure = []

            for duty in data["landedCostCalculateWorkflow"][0]["duties"]:
                if duty["item"]["id"] == item_id:
                    total_duties += duty["amount"]
                    applied_levies.append(
                        "GENERAL_DUTY"
                        if duty["description"] == "duty"
                        else duty["description"]
                    )
                    for measure in LEVY_UNITS.keys():
                        if (
                            measure in duty["formula"]
                            and LEVY_UNITS[measure] not in referenced_units_of_measure
                        ):
                            referenced_units_of_measure.append(LEVY_UNITS[measure])

            item.duty_rate = total_duties / item.item_amount
            item.applied_levies = applied_levies
            item.referenced_units_of_measure = referenced_units_of_measure

            # Get taxes for this item
            for tax in data["landedCostCalculateWorkflow"][0]["taxes"]:
                if tax["item"]["id"] == item_id:
                    total_taxes += tax["amount"]
                    applied_taxes.append(tax["description"])
                    for measure in LEVY_UNITS.keys():
                        if (
                            measure in tax["formula"]
                            and LEVY_UNITS[measure] not in referenced_units_of_measure
                        ):
                            referenced_units_of_measure.append(LEVY_UNITS[measure])

            item.tax_rate = total_taxes / item.item_amount
            item.applied_taxes = applied_taxes
            item.referenced_units_of_measure = referenced_units_of_measure

        return self.items


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    # Make a request as a test
    env = Environment("prod")
    item1 = Item(
        product_id="row1",
        country_of_origin="CA",
        hs_code="8708.99.6890",
        is_auto_part=False,
        is_usmca=False,
        contains_steel_copper_or_aluminum=False,
    )
    item2 = Item(
        product_id="row2",
        country_of_origin="CN",
        hs_code="8708.99.6890",
        is_auto_part=False,
        is_usmca=False,
        contains_steel_copper_or_aluminum=True,
    )
    landed_cost = LandedCost(
        ship_to_country="US",
        ship_from_country="CA",
        is_postal=False,
        items=[item1, item2],
    )

    items = landed_cost.process_items(env)
    print(items)
