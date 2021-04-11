from FE_Investment_Accounts.alpaca import FE_Alpaca
from setup import mount_folder
import csv, os


if __name__ == "__main__":
    alpaca = FE_Alpaca(live=True)
    list_assets = alpaca.list_assets()
    save_file = os.path.join(mount_folder, "alpaca_company_autosave.csv")
    with open(save_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Symbol", "Name", "Sector"])
        for asset in list_assets:
            writer.writerow([asset.symbol, "", ""])



