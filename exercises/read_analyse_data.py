from os import abort


class RegisterBox:
    def __init__(self, filename):
        self.data = {
            "date": [],
            "top_product": [],
            "income": [],
            "expenses": [],
            "payment_method": [],
            "revenue": []
        }

        self.number_of_days = 0
        self.total_income = 0
        self.total_revenue = 0

        self.max_income = 0
        self.min_income = 1e+10
        self.max_income_date = ""
        self.min_income_date = ""

        self.avg_revenue = 0
        self.max_revenue = 0
        self.min_revenue = 1e+10
        self.max_revenue_date = ""
        self.min_revenue_date = ""

        self.card_total = 0
        self.cash_total = 0

        self.is_file_loaded = False

        # 1. Try to open the file and load data
        try:
            with open(filename) as file:
                lines = file.readlines()
                index = 0

                # Try to convert data into the correct format

                for line in lines[1:]:
                    try:
                        line_info = line.strip().split(",")
                        self.data["date"].append(line_info[0])
                        self.data["top_product"].append(line_info[1])
                        self.data["income"].append(float(line_info[2]))
                        self.data["expenses"].append(float(line_info[3]))
                        self.data["payment_method"].append(line_info[4])
                        self.data["revenue"].append(float(line_info[2]) - float(line_info[3]))

                        self.total_income += self.data["income"][index]
                        self.total_revenue += self.data["revenue"][index]

                        if self.data["income"][index] > self.max_income:
                            self.max_income = self.data["income"][index]
                            self.max_income_date = self.data["date"][index]
                        if self.data["income"][index] < self.min_income:
                            self.min_income = self.data["income"][index]
                            self.min_income_date = self.data["date"][index]

                        if self.data["revenue"][index] > self.max_revenue:
                            self.max_revenue = self.data["revenue"][index]
                            self.max_revenue_date = self.data["date"][index]
                        if self.data["revenue"][index] < self.min_revenue:
                            self.min_revenue = self.data["revenue"][index]
                            self.min_revenue_date = self.data["date"][index]

                        if self.data["payment_method"][index] == "Tarjeta":
                            self.card_total += self.data["income"][index]
                        else:
                            self.cash_total += self.data["income"][index]

                        index += 1
                    except ValueError:
                        print(f"⚠️ Warning: Invalid data in line -> '{line.strip()}'. This line will be ignored.\n")

            self.is_file_loaded = True
            self.number_of_days = index + 1

            self.avg_revenue = self.total_revenue / self.number_of_days
            day_type = []

            for i in self.data["revenue"]:
                if i > (self.avg_revenue * 1.5):
                    day_type.append("Great")
                elif i < (self.avg_revenue * 0.8):
                    day_type.append("Bad")
                else:
                    day_type.append("Average")

            self.data["day_type"] = day_type

            print(f"✅ ¡Great! File '{filename}' was loaded correctly with {self.number_of_days} valid days.\n")

        except FileNotFoundError:
            print(f"❌ CRITIC ERROR: File '{filename}' could not be found.")
            print("   Please verify that the filepath is correct or that the file exists.")
            abort()

        except Exception as e:
            print(f"❌ An unexpected error occurred, try again please: {e}")
            abort()

    def report_day_types(self):
        day_type_count = {
            "Great": 0,
            "Bad": 0,
            "Average": 0,
        }

        for i in self.data["day_type"]:
            if i == "Great":
                day_type_count["Great"] += 1
            elif i == "Bad":
                day_type_count["Bad"] += 1
            else:
                day_type_count["Average"] += 1

        print(f"There were {day_type_count['Great']} great days, {day_type_count['Bad']} bad days and "
              f"{day_type_count['Average']} average days in this period")

    def report_top_product(self):
        product_count = {}
        for i in self.data["top_product"]:
            if i not in product_count:
                product_count[i] = 1
            else:
                product_count[i] += 1

        top_product_key = ""

        for key, value in product_count.items():
            if top_product_key == "":
                top_product_key = key
            elif value > product_count[top_product_key]:
                top_product_key = key

        print(f"{top_product_key} was the top product of the period with {product_count[top_product_key]} sells")

    def report_critic_days(self):
        i = 0
        critic_days = []
        while i < (self.number_of_days - 1):
            if self.data["expenses"][i] > 500.00 and self.data["income"][i] < 1900.00:
                critic_days.append(self.data["date"][i])
            i += 1

        print(f"There were {len(critic_days)} critic days: {', '.join(critic_days)}")

    def make_report(self):
        print(f"Days in the period: {self.number_of_days}")
        print(f"Total income: {self.total_income:.2f} ")
        print(f"Total revenue: {self.total_revenue:.2f} ")
        print(f"Average revenue: {self.avg_revenue:.2f}")

        print(f"Most profitable day {self.max_revenue_date} with revenue: {self.max_revenue:.2f}")
        print(f"Least profitable day {self.min_revenue_date} with revenue: {self.min_revenue:.2f}")

        print(f"Income by payment method: Card {self.card_total:.2f} | Cash {self.cash_total:.2f}")

        self.report_day_types()
        self.report_top_product()
        self.report_critic_days()


filepath = input("Please enter the file path: ")
register_box = RegisterBox(filepath)
register_box.make_report()
