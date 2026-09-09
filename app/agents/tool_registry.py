from google.genai import types


def get_tool_declarations():
    return [
        # =========================================================
        # CUSTOMER
        # =========================================================
        types.FunctionDeclaration(
            name="get_customer_profile",
            description=(
                "Get information about the authenticated customer's profile. "
                "Use for questions about the customer's email, mobile number, "
                "name, city, account information, registration date, purchase "
                "summary, first purchase, last purchase, or profile details. "
                "The customer identity is determined by the backend/session. "
                "Never ask the model to provide customer_id."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="ARRAY",
                        items=types.Schema(
                            type="STRING",
                            enum=[
                                "customer_code",
                                "customer_group",
                                "customer_type",
                                "city",
                                "customer_name",
                                "gender",
                                "mobile",
                                "email",
                                "created_date_persian",
                                "first_purchase",
                                "last_purchase",
                                "success_ordered_cnt",
                                "success_ordered_price",
                                "item_cnt",
                            ],
                        ),
                        description=(
                            "Specific profile fields to retrieve. "
                            "If the user asks for one specific field, "
                            "request only that field. "
                            "If the user asks for the full profile, "
                            "omit this parameter."
                        ),
                    ),
                },
            ),
        ),

        # =========================================================
        # ORDERS
        # =========================================================
        types.FunctionDeclaration(
            name="get_customer_order_summary",
            description=(
                "Get the authenticated customer's order summary. "
                "Use for total number of orders, successful orders, "
                "cancelled orders, total purchased item rows and total spending."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),

        types.FunctionDeclaration(
            name="get_latest_order",
            description=(
                "Get the authenticated customer's most recent order. "
                "Use for questions about the latest, newest, or most recent order."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={},
            ),
        ),

        types.FunctionDeclaration(
            name="get_order_status",
            description=(
                "Get the status and delivery-related information of a specific "
                "order belonging to the authenticated customer. "
                "The order_id must come from the user or from a previous "
                "customer-scoped search."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "order_id": types.Schema(
                        type="STRING",
                        description="The order ID provided by the customer.",
                    ),
                },
                required=["order_id"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_order_details",
            description=(
                "Get all item-level details for a specific order belonging "
                "to the authenticated customer."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "order_id": types.Schema(
                        type="STRING",
                        description="The order ID.",
                    ),
                },
                required=["order_id"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_order_history",
            description=(
                "Get recent orders of the authenticated customer. "
                "Use when the customer asks about previous orders, order history, "
                "recent orders, or wants a list of their orders."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "limit": types.Schema(
                        type="INTEGER",
                        description=(
                            "Maximum number of orders to return. "
                            "Maximum 50."
                        ),
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="search_customer_orders",
            description=(
                "Search ONLY the authenticated customer's orders. "
                "Can search by order ID, product name, SKU, product ID, "
                "brand, category, or order status. "
                "Never searches other customers."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "search_term": types.Schema(
                        type="STRING",
                        description=(
                            "A product name, SKU, order ID, brand, category, "
                            "or status to search for."
                        ),
                    ),
                    "limit": types.Schema(
                        type="INTEGER",
                        description=(
                            "Maximum number of matching orders. "
                            "Maximum 50."
                        ),
                    ),
                },
                required=["search_term"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_purchased_products",
            description=(
                "Get products purchased by the authenticated customer. "
                "Use when the customer asks what they bought or what products "
                "they have purchased."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "limit": types.Schema(
                        type="INTEGER",
                        description=(
                            "Maximum number of purchased item records. "
                            "Maximum 100."
                        ),
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="get_product_purchase_history",
            description=(
                "Search the authenticated customer's purchase history for a "
                "specific product, SKU, product ID, or brand. "
                "Use for questions such as whether they bought a product before, "
                "how many times they bought it, total quantity purchased, "
                "or when they last purchased it."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "product_query": types.Schema(
                        type="STRING",
                        description=(
                            "Product name, SKU, product ID, or brand to search."
                        ),
                    ),
                    "limit": types.Schema(
                        type="INTEGER",
                        description=(
                            "Maximum number of matching products. "
                            "Maximum 50."
                        ),
                    ),
                },
                required=["product_query"],
            ),
        ),
    ]