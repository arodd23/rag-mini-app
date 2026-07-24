import time

import requests


NTFY_TOPIC = "aditirod_thirdy"


def send_phone_notification(title: str, message: str) -> None:
    try:
        response = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": "high",
                "Tags": "white_check_mark",
            },
            timeout=10,
        )
        response.raise_for_status()
        print("Notification sent successfully.")

    except requests.RequestException as exc:
        print(f"Notification failed: {exc}")


if __name__ == "__main__":
    print("Test started. Waiting 30 seconds...")
    time.sleep(30)

    send_phone_notification(
        title="Delayed test complete",
        message="The 30-second test finished successfully.",
    )