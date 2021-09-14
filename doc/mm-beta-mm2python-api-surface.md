# API surface
This is a list of all of the MicroManager and mm2python API objects and methods that the acquisition script calls and/or depends on. This is specific to MicroManager 2.0 beta with the mm2python bridge and was created on 2021-09-14 (prior to the migration to gamma and pycromanager). 

## Core
```
snapImage
getTaggedImage
getCameraDevice
getShutterDevice
assignImageSynchro
setAutoShutter
setConfig
setExposure
setProperty
setPosition
waitForConfig
waitForDevice
waitForImageSynchro
waitForSystem
```

## Studio
```
mm_studio.live
mm_studio.live().snap
mm_studio.data
mm_studio.displays
mm_studio.displays().createDisplay(datastore)

datastore = mm_studio.data().createMultipageTIFFDatastore(...)
datastore.putImage
datastore.freeze

image = mm_studio.data().convertTaggedImage(mm_core.getTaggedImage())
```

## Image objects (returned by `convertTaggedImage`)
```
image.copyWith
metadata = image.getMetadata()
coords = image.getCoords()
metadata.positionName
metadata.copy
metadata.build
coords.stagePosition
coords.channel
cooords.z
coords.build
coords.copy
```

## Autofocus
```
mm_studio.getAutofocusManager
mm_studio.getAutofocusManager().setAutofocusMethodByName
mm_studio.getAutofocusManager().getAutofocusMethod
mm_studio.getAutofocusManager().getAutofocusMethod().fullFocus
mm_studio.getAutofocusManager().getAutofocusMethod().getCurrentFocusScore
```

## Position list
```
mm_studio.getPositionList
mm_studio.getPositionList().getNumberOfPositions
mm_studio.getPositionList().getPosition
mm_studio.getPositionList().getPosition(ind).getLabel
mm_studio.getPositionList().getPosition(ind).goToPosition
```

## mm2python gate
```
gate.clearQueue
meta = gate.getLastMeta
meta.getxRange
meta.getyRange
meta.getFilepath
```