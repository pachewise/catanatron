from colonist_player.colonist_enums import *
from colonist_player.utils import *

# ==== SELENIUM
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from chromedriver_py import binary_path  # this will get you the path variable

capabilities = webdriver.DesiredCapabilities.CHROME.copy()
capabilities["goog:loggingPrefs"] = {"performance": "ALL"}
driver = webdriver.Chrome(
    executable_path=binary_path, desired_capabilities=capabilities
)
driver.get("https://colonist.io/")


def consume_messages():
    messages = []
    for wsData in driver.get_log("performance"):
        # print(wsData)
        wsJson = json.loads((wsData["message"]))
        if wsJson["message"]["method"] == "Network.webSocketFrameReceived":
            message = wsJson["message"]["params"][
                "response"
            ]  # ignoring timestamp, requestId
            data = parse_websocket_message(message["payloadData"])

            print("RECEIEVED", find_in_enum(int(data["id"]), ReceivedWSType))
            print(data)
            messages.append(data)
        if wsJson["message"]["method"] == "Network.webSocketFrameSent":
            message = wsJson["message"]["params"]["response"]
            data = parse_websocket_message(message["payloadData"])
            print("SENT")
            print(data)
            messages.append(data)
    return messages


play_bots_btn = driver.find_element(By.ID, "landingpage_cta_playvsbots")
play_bots_btn.click()
print("CLICKED PLAY WITH BOTS")
while True:
    messages = consume_messages()
    driver.implicitly_wait(10)  # seconds
    canvas = driver.find_element(By.CSS_SELECTOR, "canvas")
    # i = 0
    clicks = [(0, 0), (10, 0), (20, 0), (30, 0)]
    for i in range(0, 100, 5):
        (x, y) = (i, 0 - i)
        print(i, "Clicking on", x, y)
        # driver.action.move_to(canvas, x, y).perform()
        # driver.click.perform()
        # driver.action
        drawing = (
            ActionChains(driver)
            .move_to_element_with_offset(canvas, x, y)
            .click()
            .release()
            .perform()
        )
        # drawing.perform()
        # i += 1
        # x -= 5
        # y -= 5

    print(len(messages))
    time.sleep(10)  # wait for game to load up
    # driver.implicitly_wait(10) # seconds

# Idea: collect player controller state messages
# height: 50px;
#     width: 50px;
#     background: red;
#     z-index: 1000000000;
#     position: fixed;
#     top: 160px; vs 730px. -570 to go down to actions bar.
#     left: 165px; vs 715px. -550 to go to end turn
# import asyncio
# from pyppeteer import launch

# async def main():
#     print("MAIN")
#     browser = await launch(
#         headless=False,
#         # args=["--no-sandbox"],
#         autoClose=False
#     )
#     print("Created browser")
#     page = await browser.newPage()
#     await page.goto("https://www.tradingview.com/symbols/BTCUSD/")

#     cdp = await page.target.createCDPSession()
#     await cdp.send("Network.enable")
#     await cdp.send("Page.enable")

#     def printResponse(response):
#         print(response)
#     cdp.on("Network.webSocketFrameReceived", printResponse)  # Calls printResponse when a websocket is received
#     cdp.on("Network.webSocketFrameSent", printResponse)  # Calls printResponse when a websocket is sent

#     await asyncio.sleep(100)


# asyncio.run(main())

# import asyncio
# from pyppeteer import launch

# async def main():
#     browser = await launch()
#     page = await browser.newPage()
#     await page.goto("https://example.com")
#     await page.screenshot({"path": "example.png"})
#     await browser.close()

# # asyncio.get_event_loop().run_until_complete(main())
# asyncio.run(main())
