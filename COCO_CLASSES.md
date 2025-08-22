# COCO Dataset Classes Reference

The YOLOv8 Object Detection Node can detect **80 different object classes** from the COCO dataset. Here's the complete list:

## 🧑‍🤝‍🧑 People
- **0**: person

## 🚗 Vehicles  
- **1**: bicycle
- **2**: car  
- **3**: motorcycle
- **4**: airplane
- **5**: bus
- **6**: train
- **7**: truck
- **8**: boat

## 🚦 Traffic & Outdoor
- **9**: traffic light
- **10**: fire hydrant
- **11**: stop sign
- **12**: parking meter
- **13**: bench

## 🐾 Animals
- **14**: bird
- **15**: cat
- **16**: dog
- **17**: horse
- **18**: sheep
- **19**: cow
- **20**: elephant
- **21**: bear
- **22**: zebra
- **23**: giraffe

## 🎒 Accessories & Bags
- **24**: backpack
- **25**: umbrella
- **26**: handbag
- **27**: tie
- **28**: suitcase

## ⚽ Sports & Recreation  
- **29**: frisbee
- **30**: skis
- **31**: snowboard
- **32**: sports ball
- **33**: kite
- **34**: baseball bat
- **35**: baseball glove
- **36**: skateboard
- **37**: surfboard
- **38**: tennis racket

## 🍽️ Kitchen & Dining
- **39**: bottle
- **40**: wine glass
- **41**: cup
- **42**: fork
- **43**: knife
- **44**: spoon
- **45**: bowl

## 🍎 Food
- **46**: banana
- **47**: apple
- **48**: sandwich
- **49**: orange
- **50**: broccoli
- **51**: carrot
- **52**: hot dog
- **53**: pizza
- **54**: donut
- **55**: cake

## 🏠 Furniture
- **56**: chair
- **57**: couch
- **58**: potted plant
- **59**: bed
- **60**: dining table
- **61**: toilet

## 💻 Electronics
- **62**: tv
- **63**: laptop
- **64**: mouse
- **65**: remote
- **66**: keyboard
- **67**: cell phone

## 🔌 Appliances
- **68**: microwave
- **69**: oven
- **70**: toaster
- **71**: sink
- **72**: refrigerator

## 📚 Personal Items
- **73**: book
- **74**: clock
- **75**: vase
- **76**: scissors
- **77**: teddy bear
- **78**: hair drier
- **79**: toothbrush

---

## 🎛️ Filter Controls

### Current Default Settings:
- ✅ **All 80 classes enabled** by default
- ❌ **exclude_person**: True (only excludes person)
- ❌ **exclude_animals**: False (animals included)
- ❌ **exclude_vehicles**: False (vehicles included)

### Available Filters:
- **exclude_person**: Skip person detection
- **exclude_animals**: Skip all animals (classes 14-23)  
- **exclude_vehicles**: Skip all vehicles (classes 1-8)
- **min_confidence**: Minimum confidence for any detection (0.1-0.9)

### Examples:

**Detect Everything (79 classes):**
- exclude_person: True
- exclude_animals: False  
- exclude_vehicles: False
- **Result**: All objects except people

**Objects Only (69 classes):**
- exclude_person: True
- exclude_animals: True
- exclude_vehicles: False
- **Result**: No people or animals, includes vehicles

**Pure Objects (61 classes):**
- exclude_person: True
- exclude_animals: True
- exclude_vehicles: True
- **Result**: Only furniture, electronics, food, accessories

**Everything (80 classes):**
- exclude_person: False
- exclude_animals: False
- exclude_vehicles: False
- **Result**: Detect absolutely everything including people

---

## 💡 Pro Tips:

1. **Lower confidence** (0.1-0.3) = more detections, some false positives
2. **Higher confidence** (0.5-0.8) = fewer but more accurate detections  
3. **Check console output** to see which classes are being excluded
4. **Use padding** to get more context around detected objects
5. **Adjust max_size** based on your output needs (128-1024px)

Perfect for detecting multiple objects from a rich set of 80 different classes! 🎯
