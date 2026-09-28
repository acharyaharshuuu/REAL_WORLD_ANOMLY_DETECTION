from src.events.event_detection import generate_events

def test_event_merging():
    p = [
      {"start_time":0,"end_time":2,"activity":"Assault","confidence":.8},
      {"start_time":2,"end_time":4,"activity":"Assault","confidence":.9},
      {"start_time":4,"end_time":6,"activity":"Normal","confidence":.2},
    ]
    e = generate_events(p,.7,2)
    assert len(e) == 1 and e[0]["activity"] == "Assault" and e[0]["duration"] == 4
