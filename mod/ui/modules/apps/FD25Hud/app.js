// FD-25 operator display.  Data arrives from the vehicle controller
// (lua/controller/fd25Hydraulics.lua -> guihooks.trigger("FD25HudUpdate", data)) at 10 Hz.
angular.module('beamng.apps')
.directive('fd25Hud', [function () {
  return {
    template:
      '<div class="fd25hud" style="width:100%;height:100%;box-sizing:border-box;padding:8px 10px;' +
      'background:rgba(18,20,22,0.82);border-radius:6px;border-top:3px solid #f2a20c;color:#e8e8e8;' +
      'font-family:Roboto,Arial,sans-serif;font-size:12px;overflow:hidden;">' +
      '  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">' +
      '    <span style="font-weight:700;letter-spacing:1px;color:#f2a20c;">FERROVORE FD-25</span>' +
      '    <span ng-if="d" ng-style="{background: d.mode==\'travel\' ? \'#2f7de1\' : \'#2fa84f\'}" ' +
      '          style="padding:1px 8px;border-radius:3px;font-weight:700;">{{ d.mode | uppercase }}</span>' +
      '  </div>' +
      '  <div ng-if="!d" style="opacity:0.7;padding:30px 0;text-align:center;">Waiting for an FD-25&hellip;</div>' +
      '  <div ng-if="d">' +
      '    <div style="display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;">' +
      '      <span class="chip" ng-style="chip(true)">{{ d.power | uppercase }}</span>' +
      '      <span class="chip" ng-style="chip(d.precision)">PRECISION</span>' +
      '      <span class="chip" ng-style="chip(d.level)">LEVEL</span>' +
      '      <span class="chip" ng-style="chip(d.travelHigh)">{{ d.travelHigh ? "HI" : "LO" }} RANGE</span>' +
      '      <span class="chip" ng-style="chip(d.beacon)">BEACON</span>' +
      '      <span class="chip" ng-if="d.macro" ng-style="chip(true)">AUTO {{ d.macro | uppercase }}</span>' +
      '    </div>' +
      '    <div ng-repeat="c in cyls" style="display:flex;align-items:center;margin:3px 0;">' +
      '      <span style="width:46px;opacity:0.85;">{{ c.label }}</span>' +
      '      <div style="flex:1;height:10px;background:#333;border-radius:2px;position:relative;">' +
      '        <div ng-style="{width: (d.cyl[c.key] || 0) + \'%\', background: d.cyl[c.key + \'Stall\'] ? \'#e0322a\' : \'#f2a20c\'}" ' +
      '             style="height:100%;border-radius:2px;"></div>' +
      '      </div>' +
      '      <span style="width:52px;text-align:right;" ng-style="{color: d.cyl[c.key + \'Stall\'] ? \'#ff5a4f\' : \'#ccc\'}">' +
      '        {{ d.cyl[c.key + \'Stall\'] ? \'RELIEF\' : (d.cyl[c.key] || 0) + \'%\' }}</span>' +
      '    </div>' +
      '    <div style="display:flex;justify-content:space-around;margin:6px 0 2px 0;">' +
      '      <div ng-repeat="a in angles" style="text-align:center;">' +
      '        <svg width="62" height="62" viewBox="-31 -31 62 62">' +
      '          <circle r="27" fill="#222" stroke="#555" stroke-width="2"></circle>' +
      '          <path d="M 0 -27 A 27 27 0 0 1 0 27" fill="none" stroke="#333" stroke-width="0"></path>' +
      '          <line x1="0" y1="0" x2="0" y2="-23" stroke="#f2a20c" stroke-width="3" stroke-linecap="round" ' +
      '                ng-attr-transform="rotate({{ -(d[a.key] || 0) }})"></line>' +
      '          <circle r="3" fill="#ddd"></circle>' +
      '        </svg>' +
      '        <div>{{ a.label }} {{ d[a.key] || 0 }}&deg;</div>' +
      '      </div>' +
      '      <div style="text-align:left;line-height:17px;padding-top:4px;">' +
      '        <div>Engine <b>{{ d.rpm }}</b> rpm</div>' +
      '        <div>Pump <b>{{ d.pressureMPa }}</b> MPa</div>' +
      '        <div>Flow <b>{{ d.flow }}</b> %</div>' +
      '      </div>' +
      '    </div>' +
      '    <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;">' +
      '      <button ng-repeat="b in buttons" ng-click="call(b.fn)" ' +
      '              style="flex:1 0 28%;background:#3a3d40;color:#eee;border:1px solid #555;border-radius:3px;' +
      '                     padding:3px 0;font-size:11px;cursor:pointer;">{{ b.label }}</button>' +
      '    </div>' +
      '  </div>' +
      '</div>',
    replace: true,
    restrict: 'EA',
    scope: true,
    link: function (scope, element, attrs) {
      scope.d = null;
      scope.cyls = [
        {key: 'boom', label: 'Boom'}, {key: 'stick', label: 'Stick'}, {key: 'tilt', label: 'Tilt'},
        {key: 'jaw', label: 'Jaw'}, {key: 'clamp', label: 'Clamp'}
      ];
      scope.angles = [{key: 'slewDeg', label: 'Swing'}, {key: 'rotDeg', label: 'Rotator'}];
      scope.buttons = [
        {label: 'Work/Travel', fn: 'toggleMode'}, {label: 'Precision', fn: 'togglePrecision'},
        {label: 'Power', fn: 'cyclePower'}, {label: 'Level', fn: 'toggleLevelAssist'},
        {label: 'Stow pose', fn: 'poseStow'}, {label: 'Work pose', fn: 'poseWork'},
        {label: 'Travel Hi/Lo', fn: 'toggleTravelSpeed'}, {label: 'Beacon', fn: 'toggleBeacon'}
      ];
      scope.chip = function (on) {
        return {padding: '1px 6px', borderRadius: '3px', fontSize: '10px', fontWeight: 700,
                background: on ? '#f2a20c' : '#3a3d40', color: on ? '#111' : '#999'};
      };
      scope.call = function (fn) {
        if (typeof bngApi !== 'undefined') {
          bngApi.activeObjectLua("local c = controller.getControllerSafe('fd25Hydraulics'); if c and c." + fn +
                                 " then c." + fn + "() end");
        }
      };
      var timeout = null;
      scope.$on('FD25HudUpdate', function (event, data) {
        scope.$evalAsync(function () { scope.d = data; });
        if (timeout) { clearTimeout(timeout); }
        // hide the readout again if the player switches to another vehicle
        timeout = setTimeout(function () { scope.$evalAsync(function () { scope.d = null; }); }, 1500);
      });
      scope.$on('$destroy', function () { if (timeout) { clearTimeout(timeout); } });
    }
  };
}]);
