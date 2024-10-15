│   ├── i2c@d401d800
│   │   ├── #address-cells
│   │   ├── clocks
│   │   ├── compatible
│   │   ├── cpuidle,pm-runtime,sleep
│   │   ├── interconnect-names
│   │   ├── interconnects
│   │   ├── interrupt-parent
│   │   ├── interrupts
│   │   ├── name
│   │   ├── pinctrl-0
│   │   ├── pinctrl-names
│   │   ├── power-domains
│   │   ├── reg
│   │   ├── resets
│   │   ├── #size-cells
│   │   ├── spacemit,adapter-id
│   │   ├── spacemit,apb_clock
│   │   ├── spacemit,dma-disable
│   │   ├── spacemit,i2c-clk-rate
│   │   ├── spacemit,i2c-lcr
│   │   ├── spacemit,i2c-master-code
│   │   ├── spacemit,i2c-wcr
│   │   ├── spm8821@41
│   │   │   ├── compatible
│   │   │   ├── dcdc5-supply
│   │   │   ├── interrupt-parent
│   │   │   ├── interrupts
│   │   │   ├── key
│   │   │   │   ├── compatible
│   │   │   │   └── name
│   │   │   ├── name
│   │   │   ├── pinctrl
│   │   │   │   ├── compatible
│   │   │   │   ├── #gpio-cells
│   │   │   │   ├── gpio-controller
│   │   │   │   ├── name
│   │   │   │   └── spacemit,npins
│   │   │   ├── reg
│   │   │   ├── regulators
│   │   │   │   ├── compatible
│   │   │   │   ├── DCDC_REG1
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   ├── regulator-ramp-delay
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── DCDC_REG2
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-ramp-delay
│   │   │   │   ├── DCDC_REG3
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-ramp-delay
│   │   │   │   ├── DCDC_REG4
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   ├── regulator-ramp-delay
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── DCDC_REG5
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-ramp-delay
│   │   │   │   ├── DCDC_REG6
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-ramp-delay
│   │   │   │   ├── LDO_REG1
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-boot-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG10
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   └── regulator-name
│   │   │   │   ├── LDO_REG11
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   └── regulator-name
│   │   │   │   ├── LDO_REG2
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG3
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG4
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG5
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-boot-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG6
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG7
│   │   │   │   │   ├── name
│   │   │   │   │   ├── phandle
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   ├── regulator-name
│   │   │   │   │   └── regulator-state-mem
│   │   │   │   │       ├── name
│   │   │   │   │       ├── regulator-off-in-suspend
│   │   │   │   │       └── regulator-suspend-microvolt
│   │   │   │   ├── LDO_REG8
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-always-on
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   └── regulator-name
│   │   │   │   ├── LDO_REG9
│   │   │   │   │   ├── name
│   │   │   │   │   ├── regulator-max-microvolt
│   │   │   │   │   ├── regulator-min-microvolt
│   │   │   │   │   └── regulator-name
│   │   │   │   ├── name
│   │   │   │   └── SWITCH_REG1
│   │   │   │       ├── name
│   │   │   │       └── regulator-name
│   │   │   ├── rtc
│   │   │   │   ├── compatible
│   │   │   │   └── name
│   │   │   ├── status
│   │   │   └── vcc_sys-supply
│   │   └── status

